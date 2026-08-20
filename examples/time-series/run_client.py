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

"""Script to run a federated client on the sEMG hand poses dataset."""

import os
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Union

import torch
from dataset import MaskedAutoEncoderDataset
from sklearn.model_selection import train_test_split

# Do not remove the following "unused" import,
# it is necessary for type registration
import declearn.model.torch  # noqa: F401
from declearn.communication.utils._build import NetworkClientConfig
from declearn.dataset.examples import (
    ACTIONS,
)
from declearn.dataset.torch import TorchDataset
from declearn.main._client import FederatedClient
from declearn.utils.examples import setup_client_argparse

FILEDIR = os.path.dirname(__file__)


@dataclass
class ClientConfigInput:
    """
    Configuration container for Federated Client necessary configuration.

    Fields
    ------
    folder: str
        Root folder containing the EMG data.
    actions: List[str]
        List of action/gesture labels to keep.
    target: int
        Target emg sensor index.
    subjects: List[int]
        Subject identifiers to include in the dataset.
    window_size: int
        Number of time steps per sliding window.
    zip_name: str
        Name of the archive or dataset bundle.
    on_save: bool
        Whether to save processed outputs to disk.
    on_save_filename: Optional[str]
        Filename used when saving processed data.

    """

    name: Union[str | list[str]]
    certificate: str
    data_path: str
    target: int = 8
    protocol: str = "websockets"
    server_uri: str = "wss://localhost:8765"
    verbose: bool = True


def run_client(configs: ClientConfigInput):
    """Creates and runs a client instance

    Parameters
    ----------
    configs: ClientConfigInput)
        Necessary configuration to run the client instance.
    """

    data = torch.load(configs.data_path)
    train, valid = train_test_split(data, test_size=0.20)

    train = TorchDataset(
        dataset=MaskedAutoEncoderDataset(train),
    )
    valid = TorchDataset(
        dataset=MaskedAutoEncoderDataset(valid),
    )

    name, certificate, _, _, protocol, server_uri, _ = astuple(configs)
    network = NetworkClientConfig(protocol, server_uri, name, certificate)

    client = FederatedClient(
        netwk=network,
        train_data=train,
        valid_data=valid,
    )

    client.run()


if __name__ == "__main__":
    # parse any neccessary arguments from bash using the client parser
    parser = setup_client_argparse(
        usage="Start a client providing an EMG dataset",
        default_cert=os.path.join(FILEDIR, "ca-cert.pem"),
    )
    parser.add_argument(
        type=str,
        dest="client_name",
        help="Client name. Must be the same as the name "
        "used to generate the data.",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Absolute path to the client data.",
    )
    parser.add_argument(
        "--window_size",
        default=128,
        help="Int. Sliding window length.",
    )
    parser.add_argument(
        "--actions",
        default=ACTIONS,
        help="List[str]. List of available actions from which "
        + "files are chosen to be processed",
    )
    parser.add_argument(
        "--target",
        type=int,
        help="Int. Target column to extract which represents the sensor "
        + "corresponding to the time-series.",
        default=8,
        choices=list(range(1, 9)),
    )
    args = parser.parse_args()
    data_path = os.path.abspath(Path(args.data_folder) / f"{args.name}.pt")

    # check if the data path actually exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file was not found: {data_path}")

    # set up the configs object
    client_configs = ClientConfigInput(
        name=args.name,
        data_path=data_path,
        certificate=args.certificate,
        protocol=args.protocol,
        server_uri=args.uri,
    )

    # run the client routine
    run_client(client_configs)
