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

"""Script to run a federated client on the heart-disease example."""

import os
from typing import Literal

import numpy as np

from declearn.communication import NetworkClientConfig
from declearn.dataset import InMemoryDataset
from declearn.dataset.examples import load_heart_uci
from declearn.main import FederatedClient
from declearn.test_utils import setup_client_argparse


FILEDIR = os.path.dirname(__file__)


def run_client(
    name: Literal["cleveland", "hungarian", "switzerland", "va"],
    ca_cert: str,
    protocol: str = "websockets",
    serv_uri: str = "wss://localhost:8765",
    verbose: bool = True,
) -> None:
    """Instantiate and run a given client.

    Arguments
    ---------
    name: str
        Name of the client (i.e. center data from which to use).
    ca_cert: str
        Path to the certificate authority file that was used to
        sign the server's SSL certificate.
    protocol: str, default="websockets"
        Name of the communication protocol to use.
    serv_uri: str, default="wss://localhost:8765"
        URI of the server to which to connect.
    verbose: bool, default=True
        Whether to be verbose in the displayed contents, including
        all logger information and progress bars.
    """

    # (1-2) Interface training and optional validation data.

    # Load and randomly split the dataset. Note: target is a str (column name).
    data, target = load_heart_uci(name, folder=os.path.join(FILEDIR, "data"))
    data = data.loc[np.random.permutation(data.index)]
    n_tr = round(len(data) * 0.8)  # 80% train, 20% valid

    # Wrap train and validation data as Dataset objects.
    train = InMemoryDataset(
        data=data.iloc[:n_tr],
        target=target,
        expose_classes=True,  # share unique target labels with server
    )
    valid = InMemoryDataset(
        data=data.iloc[n_tr:],
        target=target,
    )

    # (3) Define network communication parameters.

    # Here, use websockets protocol on localhost:8765, with SSL encryption.
    network = NetworkClientConfig(
        protocol=protocol,
        server_uri=serv_uri,
        name=name,
        certificate=ca_cert,
    )

    # (4) Run any necessary import statement.
    #  => None are required in this example.

    # (5) Instantiate a FederatedClient and run it.

    client = FederatedClient(
        netwk=network,
        train_data=train,
        valid_data=valid,
        checkpoint=f"{FILEDIR}/results/{name}",
        verbose=verbose,
        # Note: you may add `share_metrics=False` to prevent sending
        # evaluation metrics to the server, out of privacy concerns
    )
    client.run()


# Called when the script is called directly (using `python client.py`).
if __name__ == "__main__":
    # Parse command-line arguments.
    parser = setup_client_argparse(
        usage="Start a client providing a UCI Heart-Disease Dataset shard.",
        default_cert=os.path.join(FILEDIR, "ca-cert.pem"),
    )
    parser.add_argument(
        "name",
        type=str,
        help="name of your client",
        choices=["cleveland", "hungarian", "switzerland", "va"],
    )
    args = parser.parse_args()
    # Run the client routine.
    run_client(args.name, args.certificate, args.protocol, args.uri)
