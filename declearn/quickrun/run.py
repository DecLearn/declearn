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

"""
Script to quickly run example locally using declearn.

The script requires to be provided with the path to a folder containing:

* A declearn model
* A TOML file with all the elements required to configurate an FL experiment
* A data folder, structured in a specific way

If not provided with this, the script defaults to the MNIST example provided
by declearn in `declearn.example.quickrun`.

The script then locally runs the FL experiment as layed out in the TOML file,
using privided model and data, and stores its result in the same folder.
"""

import argparse
import importlib
import logging
import os
import re
import textwrap
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.dataset import InMemoryDataset
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.model.api import Model
from declearn.quickrun._config import (
    DataSplitConfig,
    ExperimentConfig,
    ModelConfig,
)
from declearn.quickrun._parser import parse_data_folder
from declearn.quickrun._split_data import split_data
from declearn.test_utils import make_importable
from declearn.utils import (
    LOGGING_LEVEL_MAJOR,
    get_logger,
    run_as_processes,
    set_device_policy,
)

__all__ = ["quickrun"]


DEFAULT_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "../../examples/quickrun",
)


def get_model(folder, model_config) -> Model:
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
    paths: dict,
) -> None:
    """Routine to run a FL client, called by `run_declearn_experiment`."""
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
    client = FederatedClient(network, train, valid, checkpoint)
    client.run()


def get_toml_folder(config: Optional[str] = None) -> Tuple[str, str]:
    """Deternmine if provided config is a file or a directory, and
    return :
    * The path to the TOML config file
    * The path to the main folder of the experiment
    """
    # default to the mnist example
    if not config:
        config = DEFAULT_FOLDER
    config = os.path.abspath(config)
    # check if config is TOML or dir
    if os.path.isfile(config):
        toml = config
        folder = os.path.dirname(config)
    elif os.path.isdir(config):
        folder = config
        toml = f"{folder}/config.toml"
    else:
        raise FileNotFoundError(
            f"Failed to find quickrun config file at '{config}'."
        )
    return toml, folder


def locate_or_create_split_data(toml: str, folder: str) -> Dict:
    """Attempts to find split data according to the config toml or
    or the default behavior. If failed, attempts to find full data
    according to the config toml and split it"""
    data_config = DataSplitConfig.from_toml(toml, False, "data")
    try:
        client_dict = parse_data_folder(data_config, folder)
    except ValueError:
        split_data(data_config, folder)
        client_dict = parse_data_folder(data_config, folder)
    return client_dict


def server_to_client_network(
    network_cfg: NetworkServerConfig,
) -> NetworkClientConfig:
    "Converts server network config to client network config"
    return NetworkClientConfig.from_params(
        protocol=network_cfg.protocol,
        server_uri=f"ws://localhost:{network_cfg.port}",
        name="replaceme",
    )


def quickrun(config: Optional[str] = None) -> None:
    """Run a server and its clients using multiprocessing.

    The kwargs are the arguments expected by split_data,
    see [the documentation][declearn.quickrun._split_data]
    """
    toml, folder = get_toml_folder(config)
    # locate split data or split it if needed
    client_dict = locate_or_create_split_data(toml, folder)
    # Parse toml files
    ntk_server_cfg = NetworkServerConfig.from_toml(toml, False, "network")
    ntk_client_cfg = server_to_client_network(ntk_server_cfg)
    optim_cgf = FLOptimConfig.from_toml(toml, False, "optim")
    run_cfg = FLRunConfig.from_toml(toml, False, "run")
    model_cfg = ModelConfig.from_toml(toml, False, "model")
    expe_cfg = ExperimentConfig.from_toml(toml, False, "experiment")
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


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Set up and run a command-line arguments parser."""
    usage = """
        Quickly run an example locally using declearn.
        The script requires to be provided with the path a TOML file
        with all the elements required to configurate an FL experiment,
        or the path to a folder containing :
        * a TOML file with all the elements required to configurate an
        FL experiment
        * A declearn model
        * A data folder, structured in a specific way

        If not provided with this, the script defaults to the MNIST example
        provided by declearn in `declearn.example.quickrun`.
    """
    usage = re.sub("\n *(?=[a-z])", " ", textwrap.dedent(usage))
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        usage=re.sub("- ", "-", usage),
    )
    parser.add_argument(
        "--config",
        default=None,
        dest="config",
        help="Path to the root folder where to export data.",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> None:
    """Quikcrun based on commandline-input arguments."""
    cmdargs = parse_args(args)
    quickrun(config=cmdargs.config)


if __name__ == "__main__":
    main()
