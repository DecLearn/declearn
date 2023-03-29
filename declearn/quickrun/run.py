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
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.dataset import InMemoryDataset
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.model.api import Model
from declearn.test_utils import make_importable
from declearn.utils import run_as_processes

__all__ = ["quickrun"]

DEFAULT_FOLDER = "./examples/quickrun"

# Perform local imports.
# pylint: disable=wrong-import-order, wrong-import-position
with make_importable(os.path.dirname(__file__)):
    from _config import DataSplitConfig, ExperimentConfig, ModelConfig
    from _parser import parse_data_folder
    from _split_data import split_data
# pylint: enable=wrong-import-order, wrong-import-position


def _get_model(folder, model_config) -> Model:
    file = "model"
    if m_file := model_config.model_file:
        folder = os.path.dirname(m_file)
        file = m_file.rsplit("/", 1)[-1].split(".")[0]
    with make_importable(folder):
        mod = importlib.import_module(file)
        model_cls = getattr(mod, model_config.model_name)
    return model_cls


def _run_server(
    folder: str,
    network: NetworkServerConfig,
    model_config: ModelConfig,
    optim: FLOptimConfig,
    config: FLRunConfig,
) -> None:
    """Routine to run a FL server, called by `run_declearn_experiment`."""
    model = _get_model(folder, model_config)
    server = FederatedServer(model, network, optim)
    server.run(config)


def _run_client(
    folder: str,
    network: NetworkClientConfig,
    model_config: ModelConfig,
    name: str,
    paths: dict,
) -> None:
    """Routine to run a FL client, called by `run_declearn_experiment`."""
    # Overwrite client name based on folder name
    network.name = name
    # Make the model importable
    _ = _get_model(folder, model_config)
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
    client = FederatedClient(network, train, valid)
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
    return toml, folder


def locate_or_create_split_data(toml: str, folder: str) -> Dict:
    """Attempts to find split data according to the config toml or
    or the defualt behavior. If failed, attempts to find full data
    according to the config toml and split it"""
    expe_config = ExperimentConfig.from_toml(toml, False, "experiment")
    try:
        client_dict = parse_data_folder(expe_config, folder)
    except ValueError:
        data_config = DataSplitConfig.from_toml(toml, False, "data")
        split_data(folder, data_config)
        client_dict = parse_data_folder(expe_config,folder)
    return client_dict


def quickrun(
    config: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Run a server and its clients using multiprocessing.

    The kwargs are the arguments expected by split_data,
    see [the documentation][declearn.quickrun._split_data]
    """
    toml, folder = get_toml_folder(config)
    # locate split data or split it if needed
    client_dict = locate_or_create_split_data(toml, folder)
    # Parse toml file to ServerConfig and ClientConfig
    ntk_server = NetworkServerConfig.from_toml(toml, False, "network_server")
    optim = FLOptimConfig.from_toml(toml, False, "optim")
    run = FLRunConfig.from_toml(toml, False, "run")
    ntk_client = NetworkClientConfig.from_toml(toml, False, "network_client")
    model_config = ModelConfig.from_toml(toml, False, "model")
    # Set up a (func, args) tuple specifying the server process.
    p_server = (_run_server, (folder, ntk_server, model_config, optim, run))
    # Set up the (func, args) tuples specifying client-wise processes.
    p_client = []
    for name, data_dict in client_dict.items():
        client = (
            _run_client,
            (folder, ntk_client, model_config, name, data_dict),
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
    quickrun(folder=cmdargs.config)


if __name__ == "__main__":
    main()
