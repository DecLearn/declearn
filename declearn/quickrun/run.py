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
from typing import Any, Dict, List, Optional

from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.dataset import InMemoryDataset
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.test_utils import make_importable
from declearn.utils import run_as_processes

__all__ = ["quickrun"]

DEFAULT_FOLDER = "./examples/quickrun"

# Perform local imports.
# pylint: disable=wrong-import-order, wrong-import-position
with make_importable(os.path.dirname(__file__)):
    from _split_data import split_data
# pylint: enable=wrong-import-order, wrong-import-position


def _run_server(
    folder: str,
    network: NetworkServerConfig,
    optim: FLOptimConfig,
    config: FLRunConfig,
) -> None:
    """Routine to run a FL server, called by `run_declearn_experiment`."""
    # get Model
    name = "MyModel"
    with make_importable(folder):
        mod = importlib.import_module("model")
        model_cls = getattr(mod, name)
        model = model_cls
    server = FederatedServer(model, network, optim)
    server.run(config)


def parse_data_folder(folder: str) -> Dict:
    """Utils parsing a data folder following a standard format into a nested"
    dictionnary.

    The expected format is :

        folder/
        └─── data*/
            └─── client*/
            │      train_data.* - training data
            │      train_target.* - training labels
            │      valid_data.* - validation data
            │      valid_target.* - validation labels
            └─── client*/
            │    ...
    """
    # Get data dir
    gen_folders = Path(folder).glob("data*")
    data_folder = next(gen_folders, False)
    if not data_folder:
        raise ValueError(
            f"No folder starting with 'data' found in {folder}. "
            "Please store your data under a 'data_*' folder"
        )
    if next(gen_folders, False):
        raise ValueError(
            "More than one folder starting with 'data' found"
            f"in {folder}. Please store your data under a single"
            "parent folder"
        )
    # Get clients dir
    gen_folders = data_folder.glob("client*")  # type: ignore
    first_client = next(gen_folders, False)
    if not first_client:
        raise ValueError(
            f"No folder starting with 'client' found in {data_folder}. "
            "Please store your individual under client data under"
            "a 'client*' folder"
        )
    clients = {str(first_client).rsplit("/", 1)[-1]: {}}
    while client := next(gen_folders, False):
        clients[str(client).rsplit("/", 1)[-1]] = {}
    # Get train and valid files
    data_items = [
        "train_data",
        "train_target",
        "valid_data",
        "valid_target",
    ]
    for client, files in clients.items():
        for d in data_items:
            gen_file = Path(data_folder / client).glob(f"{d}*")  # type: ignore
            file = next(gen_file, False)
            if not file:
                raise ValueError(
                    f"Could not find a file named '{d}.*' in {client}"
                )
            if next(gen_file, False):
                raise ValueError(
                    f"Found more than one file named '{d}.*' in {client}"
                )
            files[d] = str(file)

    return clients


def _run_client(
    network: NetworkClientConfig,
    name: str,
    paths: dict,
    folder: str,
) -> None:
    """Routine to run a FL client, called by `run_declearn_experiment`."""
    # Overwrite client name based on folder name
    network.name = name
    # Wrap train and validation data as Dataset objects.
    name = "MyModel"
    with make_importable(folder):
        mod = importlib.import_module("model")
        model_cls = getattr(mod, name)  # pylint: disable=unused-variable
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


def quickrun(
    folder: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Run a server and its clients using multiprocessing.

    The kwargs are the arguments expected by split_data,
    see [the documentation][declearn.quickrun._split_data]
    """
    # default to the mnist exampl
    if not folder:
        folder = DEFAULT_FOLDER
    folder = os.path.abspath(folder)
    # Get datasets and client_names from folder
    try:
        client_dict = parse_data_folder(folder)
    except ValueError:
        split_data(folder, **kwargs)
        client_dict = parse_data_folder(folder)
    # Parse toml file to ServerConfig and ClientConfig
    toml = f"{folder}/config.toml"
    ntk_server = NetworkServerConfig.from_toml(toml, False, "network_server")
    optim = FLOptimConfig.from_toml(toml, False, "optim")
    run = FLRunConfig.from_toml(toml, False, "run")
    ntk_client = NetworkClientConfig.from_toml(toml, False, "network_client")
    # Set up a (func, args) tuple specifying the server process.
    p_server = (_run_server, (folder, ntk_server, optim, run))
    # Set up the (func, args) tuples specifying client-wise processes.
    p_client = []
    for name, data_dict in client_dict.items():
        client = (_run_client, (ntk_client, name, data_dict, folder))
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
        The script requires to be provided with the path to a folder
        containing:
        * A declearn model
        * A TOML file with all the elements required to configurate an FL
        experiment
        * A data folder, structured in a specific way

        If not provided with this, the script defaults to the MNIST example
        provided by declearn in `declearn.example.quickrun`.

        Once launched, this script splits data into heterogeneous shards. It
        then locally runs the FL experiment as layed out in the TOML file,
        using privided model and data, and stores its result in the same folder.

        The implemented schemes are the following:
        * "iid":
            Split the dataset through iid random sampling.
        * "labels":
            Split the dataset into shards that hold all samples
            that have mutually-exclusive target classes.
        * "biased":
            Split the dataset through random sampling according
            to a shard-specific random labels distribution.
    """
    usage = re.sub("\n *(?=[a-z])", " ", textwrap.dedent(usage))
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        usage=re.sub("- ", "-", usage),
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=5,
        help="Number of shards between which to split the data.",
    )
    parser.add_argument(
        "--root",
        default=None,
        dest="folder",
        help="Path to the root folder where to export data.",
    )
    parser.add_argument(
        "--data_path",
        default=None,
        dest="data",
        help="Path to the data to be split",
    )
    parser.add_argument(
        "--target_path",
        default=None,
        dest="target",
        help="Path to the labels to be split",
    )
    schemes_help = """
        Splitting scheme(s) to use, among {"iid", "labels", "biased"}.
        If this argument is not specified, all "iid" is used.
        See details above on the schemes' definition.
    """
    parser.add_argument(
        "--scheme",
        action="append",
        choices=["iid", "labels", "biased"],
        default=["iid"],
        dest="schemes",
        nargs="+",
        help=textwrap.dedent(schemes_help),
    )
    parser.add_argument(
        "--train_split",
        default=0.8,
        dest="perc_train",
        type=float,
        help="What proportion of the data to use for training vs validation",
    )
    parser.add_argument(
        "--seed",
        default=20221109,
        dest="seed",
        type=int,
        help="RNG seed to use (default: 20221109).",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> None:
    """Quikcrun based on commandline-input arguments."""
    cmdargs = parse_args(args)
    for scheme in cmdargs.schemes:
        quickrun(
            folder=cmdargs.folder,
            n_shards=cmdargs.n_shards,
            data=cmdargs.data,
            target=cmdargs.target,
            scheme=scheme,
            perc_train=cmdargs.perc_train,
            seed=cmdargs.seed,
        )


if __name__ == "__main__":
    main()
