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

"""TODO"""
import importlib
from glob import glob

from declearn.communication import NetworkClientConfig, NetworkServerConfig
from declearn.dataset import InMemoryDataset
from declearn.main import FederatedClient, FederatedServer
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.test_utils import make_importable
from declearn.utils import run_as_processes

DEFAULT_FOLDER = "./examples/quickrun"


def _run_server(
    model: str,
    network: NetworkServerConfig,
    optim: FLOptimConfig,
    config: FLRunConfig,
) -> None:
    """Routine to run a FL server, called by `run_declearn_experiment`."""
    server = FederatedServer(model, network, optim)
    server.run(config)


def _parse_data_folder(folder: str):
    """Utils parsing a data folder following a standard format into a nested"
    dictionnary"""
    # Get data dir
    data_folder = glob("data_*", root_dir=folder)
    if len(data_folder) == 0:
        raise ValueError(
            f"No folder starting with 'data_' found in {folder}"
            "Please store your data under a 'data_*' folder"
        )
    if len(data_folder) > 1:
        raise ValueError(
            "More than one folder starting with 'data_' found"
            f"in {folder}. Please store your data under a single"
            "parent folder"
        )
    data_folder = f"{folder}/{data_folder[0]}"
    # Get clients dir
    clients_folders = glob("client_*", root_dir=data_folder)
    if len(clients_folders) == 0:
        raise ValueError(
            f"No folder starting with 'client_' found in {data_folder}"
            "Please store your individual under client data under"
            "a 'client_*' folder"
        )
    clients = {c: {} for c in clients_folders}
    # Get train and valid files
    for c in clients.keys():
        path = f"{data_folder}/{c}/"
        data_items = [
            "train_data",
            "train_target",
            "valid_data",
            "valid_target",
        ]
        for d in data_items:
            files = glob(f"{d}*", root_dir=path)
            if len(files) != 1:
                raise ValueError(
                    f"Could not find unique file named '{d}.*' in {path}"
                )
            clients[c][d] = files[0]

    return clients


def _run_client(
    network: str,
    name: str,
    paths: dict,
) -> None:
    """Routine to run a FL client, called by `run_declearn_experiment`."""
    # Run the declearn FL client routine.
    netwk = NetworkClientConfig.from_toml(network)
    # Overwrite client name based on folder name
    netwk.name = name
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
    client = FederatedClient(netwk, train, valid)
    client.run()


def quickrun(
    folder: str = None,
) -> None:
    """Run a server and its clients using multiprocessing."""
    # default to the 101 example
    if not folder:
        folder = DEFAULT_FOLDER  # TODO check data was run
    # Parse toml file to ServerConfig and ClientConfig
    toml = f"{folder}/config.toml"
    ntk_server = NetworkServerConfig.from_toml(toml, False, "network_server")
    optim = FLOptimConfig.from_toml(toml, False, "optim")
    run = FLRunConfig.from_toml(toml, False, "run")
    ntk_client = NetworkClientConfig.from_toml(toml, False, "network_client")
    # get Model
    module, name = f"{folder}/model.py", "MyModel"
    mod = importlib.import_module(module)
    model_cls = getattr(mod, name)
    model = model_cls()
    # Set up a (func, args) tuple specifying the server process.
    p_server = (_run_server, (model, ntk_server, optim, run))
    # Get datasets and client_names from folder
    client_dict = _parse_data_folder(folder)
    # Set up the (func, args) tuples specifying client-wise processes.
    p_client = []
    for name, data_dict in client_dict.items():
        client = (_run_client, (ntk_client, name, data_dict))
        p_client.append(client)
    # Run each and every process in parallel.
    success, outputs = run_as_processes(p_server, *p_client)
    assert success, "The FL process failed:\n" + "\n".join(
        str(exc) for exc in outputs if isinstance(exc, RuntimeError)
    )


if __name__ == "__main__":
    quickrun()
