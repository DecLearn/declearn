# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Script to run a federated server on the Hand Poses sEMG example."""

import datetime
import logging
import os
from dataclasses import astuple, dataclass

from model import ModelConfigsInput, SimpleMaskedTSAutoEncoder
from torch.nn import MSELoss

from declearn.communication.utils import NetworkServerConfig
from declearn.main import FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.model.torch import TorchModel
from declearn.utils import setup_root_logger, setup_server_loggers
from declearn.utils.examples import setup_server_argparse


@dataclass
class ServerConfigInput:
    """Server input configuration container.

    Fields
    ------
    certificate: str
        Path to the client-required CA certificate PEM file.
    private_key: str
        Path to the server's private key PEM file.
    nb_clients: int
        Number of clients to await.
    protocol: str
        Expected communication protocol.
    host: str
        Expected hosting server.
    port: str
        Running port.

    """

    certificate: str
    private_key: str
    nb_clients: int = 2
    protocol: str = "websockets"
    host: str = "localhost"
    port: int = 8765


FILEDIR = os.path.dirname(os.path.abspath(__file__))


def run_server(
    server_configs: ServerConfigInput, model_configs: ModelConfigsInput
):
    """Runs a server with the defined configurations"""
    model = TorchModel(
        model=SimpleMaskedTSAutoEncoder(model_configs), loss=MSELoss()
    )
    # Set up checkpointing and logging.
    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    checkpoint = os.path.join(FILEDIR, f"result_{stamp}", "server")

    setup_server_loggers(
        level=logging.INFO, fpath=os.path.join(checkpoint, "logs.txt")
    )

    certificate, private_key, nb_clients, protocol, host, port = astuple(
        server_configs
    )
    aggregator = {"name": "averaging", "steps_weighted": True}

    client_opt = {
        "lrate": 0.02,
        "modules": ["rmsprop"],
    }
    server_opt = {
        "lrate": 1.0,
        "modules": [("momentum", {"beta": 0.95})],
    }
    # Wrap this up into an OptimizationStrategy object.
    optim = FLOptimConfig.from_params(
        aggregator=aggregator,
        client_opt=client_opt,
        server_opt=server_opt,
    )
    network = NetworkServerConfig(
        protocol=protocol,
        host=host,
        port=port,
        certificate=certificate,
        private_key=private_key,
    )
    server = FederatedServer(
        # fmt: off
        model,
        network,
        optim,
        metrics=["binary-classif", "binary-roc"],
        checkpoint=f"{FILEDIR}/results/server",
    )
    run_cfg = FLRunConfig.from_params(
        rounds=20,
        register={"min_clients": nb_clients},
        training={"batch_size": 30, "drop_remainder": False},
        evaluate={"batch_size": 50, "drop_remainder": False},
        early_stop={"tolerance": 0.0, "patience": 5, "relative": False},
    )
    server.run(run_cfg)


# Called when the script is called directly (using `python server.py`).
if __name__ == "__main__":
    # Parse command-line arguments.
    parser = setup_server_argparse(
        usage="Start a server to train a simple time-series Auto-encoder",
        default_cert=os.path.join(FILEDIR, "server-cert.pem"),
        default_pkey=os.path.join(FILEDIR, "server-pkey.pem"),
    )
    parser.add_argument(
        dest="nb_clients",
        type=int,
        help="Int. Number of clients for the experiment must be in [1-8]",
        default=2,
    )
    parser.add_argument(
        "--window_size",
        default=128,
        help="Int. Sliding window length. Default to 128",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        help="Float. Percentage of random mask applied on every"
        + "extracted window from the signal",
        choices=[0.25, 0.50, 0.75, 0.90],
        default=0.25,
    )

    args = parser.parse_args()

    server_config = ServerConfigInput(
        nb_clients=args.nb_clients,
        certificate=args.certificate,
        private_key=args.private_key,
    )
    model_configs = ModelConfigsInput(
        input_dim=args.window_size, mask_ratio=args.mask_ratio
    )

    setup_root_logger()  # to display all info logs in the console

    # Run the server routine.
    run_server(server_config, model_configs)
