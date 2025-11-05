# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
# et Automatique)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to run a federated server on the HuggingFace IMDb dataset example."""

import datetime
import os

import fire
from torch.nn import CrossEntropyLoss

import declearn
from declearn.metrics import MulticlassAccuracyPrecisionRecall
from declearn.model.torch import TorchModel
from declearn.test_utils import make_importable

# Perform local imports.
with make_importable(os.path.dirname(__file__)):
    from model import CustomDistilBertClassif

FILEDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CERT = os.path.join(FILEDIR, "server-cert.pem")
DEFAULT_PKEY = os.path.join(FILEDIR, "server-pkey.pem")


def run_server(
    nb_clients: int,
    certificate: str = DEFAULT_CERT,
    private_key: str = DEFAULT_PKEY,
    protocol: str = "websockets",
    host: str = "localhost",
    port: int = 8765,
) -> None:
    """Instantiate and run the orchestrating server.

    Arguments
    ---------
    nb_clients: int
        Exact number of clients used in this example.
    certificate: str
        Path to the (self-signed) SSL certificate to use.
    private_key: str
        Path to the associated private-key to use.
    protocol: str, default="websockets"
        Name of the communication protocol to use.
    host: str, default="localhost"
        Hostname or IP address on which to serve.
    port: int, default=8765
        Communication port on which to serve.
    """

    ### Optional: some convenience settings

    # Set GPU as device (if no GPU available, use CPU but show a warning)
    # Note : GPU run is recommended for this example (to save time)
    declearn.utils.set_device_policy(gpu=True)

    # Set up metrics suitable for IMDb task.
    metrics = declearn.metrics.MetricSet(
        [MulticlassAccuracyPrecisionRecall(labels=[0, 1])]
    )

    # Set up checkpointing and logging.
    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    checkpoint = os.path.join(FILEDIR, f"result_{stamp}", "server")
    # Set up a logger, records from which will go to a file.
    logger = declearn.utils.get_logger(
        name="Server",
        fpath=os.path.join(checkpoint, "logs.txt"),
    )

    ### (1) Define a model

    # Here we use a wrapped "small" HuggingFace transformer
    model = TorchModel(
        model=CustomDistilBertClassif(), loss=CrossEntropyLoss()
    )

    ### (2) Define an optimization strategy

    # Set up the cient updates' aggregator. By default: FedAvg.
    aggregator = declearn.aggregator.AveragingAggregator()

    # Set up the server-side optimizer (to refine aggregated updates).
    # By default: no refinement (lrate=1.0, no plug-ins).
    server_opt = declearn.optimizer.Optimizer(
        lrate=1.0,
        w_decay=0.0,
        modules=None,
    )

    # Set up the client-side optimizer (for local SGD steps).
    # By default: vanilla SGD, with a selected learning rate.
    client_opt = declearn.optimizer.Optimizer(
        lrate=0.001,
        w_decay=0.0,
        regularizers=None,
        modules=None,
    )

    # Wrap all this into a FLOptimConfig.
    optim = declearn.main.config.FLOptimConfig.from_params(
        aggregator=aggregator,
        server_opt=server_opt,
        client_opt=client_opt,
    )

    ### (3) Define network communication parameters.

    # Use user-provided parameters (or default WSS on localhost:8765).
    network = declearn.communication.build_server(
        protocol=protocol,
        host=host,
        port=port,
        certificate=certificate,
        private_key=private_key,
    )

    ### (4) Instantiate and run a FederatedServer.

    # Instanciate
    server = declearn.main.FederatedServer(
        model=model,
        netwk=network,
        optim=optim,
        metrics=metrics,
        checkpoint=checkpoint,
        logger=logger,
    )

    # Set up the experiment's hyper-parameters.
    # Registration rules: wait for exactly `nb_clients`, at most 5 minutes.
    register = declearn.main.config.RegisterConfig(
        min_clients=nb_clients,
        max_clients=nb_clients,
        timeout=300,
    )
    # Training rounds hyper-parameters. By default, 1 epoch / round.
    training = declearn.main.config.TrainingConfig(
        batch_size=32,
        n_epoch=1,
    )
    # Evaluation rounds. By default, 1 epoch with train's batch size.
    evaluate = declearn.main.config.EvaluateConfig(
        batch_size=128,
    )
    # Wrap all this into a FLRunConfig.
    run_config = declearn.main.config.FLRunConfig.from_params(
        rounds=3,  # you may change the number of training rounds
        register=register,
        training=training,
        evaluate=evaluate,
        privacy=None,  # you may set up local DP (DP-SGD) here
        early_stop=None,  # you may add an early-stopping criterion here
    )
    server.run(run_config)


# This part should not be altered: it provides with an argument parser.
# for `python server.py`.


def main():
    "Fire-wrapped `run_server`."
    fire.Fire(run_server)


if __name__ == "__main__":
    main()
