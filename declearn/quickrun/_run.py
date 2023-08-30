# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""Script to quickly run a simulated FL example locally using declearn.

The script requires to be provided with the path to a folder containing:

* A python file in which a declearn model is instantiated (in main scope).
* A TOML file with all the elements required to configure an FL experiment.
* A data folder, structured in a specific way.

If not provided with this, the script defaults to the MNIST example provided
by declearn in `declearn.example.quickrun`.

The script then locally runs the FL experiment as layed out in the TOML file,
using privided model and data, and stores its result in the same folder.
"""

import importlib
import logging
import os
from datetime import datetime
from typing import Dict, Tuple

import fire  # type: ignore

from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.dataset import InMemoryDataset
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.model.api import Model
from declearn.quickrun._config import (
    DataSourceConfig,
    ExperimentConfig,
    ModelConfig,
)
from declearn.quickrun._parser import parse_data_folder
from declearn.test_utils import make_importable
from declearn.utils import (
    LOGGING_LEVEL_MAJOR,
    get_logger,
    run_as_processes,
    set_device_policy,
)

__all__ = ["quickrun"]


def get_model(folder: str, model_config: ModelConfig) -> Model:
    "Return a model instance from a model config instance"
    path = model_config.model_file or os.path.join(folder, "model.py")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("Model file not found: '{path}'.")
    with make_importable(os.path.dirname(path)):
        mod = importlib.import_module(os.path.basename(path)[:-3])
        model = getattr(mod, model_config.model_name)
    return model


def get_checkpoint(folder: str, expe_config: ExperimentConfig) -> str:
    """Return the checkpoint folder, either default or as given in config"""
    if expe_config.checkpoint:
        checkpoint = expe_config.checkpoint
    else:
        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
        checkpoint = os.path.join(folder, f"result_{timestamp}")
    return checkpoint


def run_server(
    folder: str,
    network: NetworkServerConfig,
    model_config: ModelConfig,
    optim: FLOptimConfig,
    config: FLRunConfig,
    expe_config: ExperimentConfig,
) -> None:
    """Routine to run a FL server, called by `run_declearn_experiment`."""
    # arguments serve modularity; pylint: disable=too-many-arguments
    set_device_policy(gpu=False)
    model = get_model(folder, model_config)
    checkpoint = get_checkpoint(folder, expe_config)
    checkpoint = os.path.join(checkpoint, "server")
    logger = get_logger("Server", fpath=os.path.join(checkpoint, "logger.txt"))
    server = FederatedServer(
        model, network, optim, expe_config.metrics, checkpoint, logger
    )
    server.run(config)


def run_client(
    folder: str,
    network: NetworkClientConfig,
    model_config: ModelConfig,
    expe_config: ExperimentConfig,
    name: str,
    paths: Dict[str, str],
) -> None:
    """Routine to run a FL client, called by `run_declearn_experiment`."""
    # arguments serve modularity; pylint: disable=too-many-arguments
    # Overwrite client name based on folder name.
    network.name = name
    # Make the model importable and disable GPU use.
    set_device_policy(gpu=False)
    _ = get_model(folder, model_config)
    # Add checkpointer.
    checkpoint = get_checkpoint(folder, expe_config)
    checkpoint = os.path.join(checkpoint, name)
    # Set up a logger: write everything to file, but filter console outputs.
    logger = get_logger(name, fpath=os.path.join(checkpoint, "logs.txt"))
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(LOGGING_LEVEL_MAJOR)
    # Wrap train and validation data as Dataset objects.
    train = InMemoryDataset(
        paths.get("train_data"),
        target=paths.get("train_target"),
        expose_classes=True,
    )
    valid = InMemoryDataset(
        paths.get("valid_data"),
        target=paths.get("valid_target"),
    )
    client = FederatedClient(network, train, valid, checkpoint, logger=logger)
    client.run()


def get_toml_folder(config: str) -> Tuple[str, str]:
    """Return the path to an experiment's folder and TOML config file.

    Parameters
    ----------
    config: str
        Path to either a TOML config file (within an experiment folder),
        or to the experiment folder containing a "config.toml" file.

    Returns
    -------
    toml:
        The path to the TOML config file.
    folder:
        The path to the main folder of the experiment.

    Raises
    ------
    FileNotFoundError:
        If the TOML config file cannot be found based on inputs.
    """
    config = os.path.abspath(config)
    if os.path.isdir(config):
        folder = config
        toml = os.path.join(folder, "config.toml")
    else:
        toml = config
        folder = os.path.dirname(toml)
    if not os.path.isfile(toml):
        raise FileNotFoundError(
            f"Failed to find quickrun config file at '{config}'."
        )
    return toml, folder


def locate_split_data(toml: str, folder: str) -> Dict:
    """Attempt to find split data according to the config toml or default."""
    data_config = DataSourceConfig.from_toml(toml, False, "data")
    client_dict = parse_data_folder(data_config, folder)
    return client_dict


def server_to_client_network(
    network_cfg: NetworkServerConfig,
) -> NetworkClientConfig:
    "Convert server network config to client network config."
    return NetworkClientConfig.from_params(
        protocol=network_cfg.protocol,
        server_uri=network_cfg.build_server().uri,
        name="replaceme",
    )


def quickrun(config: str) -> None:
    """Run a server and its clients using multiprocessing.

    The script requires to be provided with the path to a TOML file
    with all the elements required to configurate an FL experiment,
    or the path to a folder containing :

    * A TOML file with all the elements required to configure an FL experiment.
    * A python file in which a declearn model is instantiated (in main scope).
    * A data folder, structured in a specific way:
        folder/
            [client_a]/
                train_data.(csv|npy|sparse|svmlight)
                train_target.(csv|npy|sparse|svmlight)
                valid_data.(csv|npy|sparse|svmlight)
                valid_target.(csv|npy|sparse|svmlight)
            [client_b]/
                ...
            ...

    Parameters
    ----------
    config: str
        Path to either a toml file or a properly formatted folder
        containing the elements required to launch the experiment.

    Notes
    -----
    - The data folder structure may be obtained by using the `declearn-split`
      commandline entry-point, or the `declearn.dataset.split_data` util.
    - The quickrun mode works by simulating a federated learning process, where
      all clients operate under parallel python processes, and communicate over
      the localhost using un-encrypted websockets communications.
    - When run without any argument, this script/function operates on a basic
      MNIST example, for demonstration purposes.
    - You may refer to a more detailed MNIST example on our GitLab repository.
      See the `examples/mnist_quickrun` folder.
    """
    # main script; pylint: disable=too-many-locals
    toml, folder = get_toml_folder(config)
    # locate split data or split it if needed
    client_dict = locate_split_data(toml, folder)
    # Parse toml files
    ntk_server_cfg = NetworkServerConfig.from_toml(toml, False, "network")
    ntk_client_cfg = server_to_client_network(ntk_server_cfg)
    optim_cgf = FLOptimConfig.from_toml(toml, False, "optim")
    run_cfg = FLRunConfig.from_toml(toml, False, "run")
    model_cfg = ModelConfig.from_toml(toml, False, "model", True)
    expe_cfg = ExperimentConfig.from_toml(toml, False, "experiment", True)
    # Set up a (func, args) tuple specifying the server process.
    p_server = (
        run_server,
        (folder, ntk_server_cfg, model_cfg, optim_cgf, run_cfg, expe_cfg),
    )
    # Set up the (func, args) tuples specifying client-wise processes.
    p_client = []
    for name, data_dict in client_dict.items():
        client = (
            run_client,
            (folder, ntk_client_cfg, model_cfg, expe_cfg, name, data_dict),
        )
        p_client.append(client)
    # Run each and every process in parallel.
    success, outputs = run_as_processes(p_server, *p_client)
    assert success, "The FL process failed:\n" + "\n".join(
        str(exc) for exc in outputs if isinstance(exc, RuntimeError)
    )


def main() -> None:
    """Fire-wrapped `quickrun`."""
    fire.Fire(quickrun)


if __name__ == "__main__":
    main()
