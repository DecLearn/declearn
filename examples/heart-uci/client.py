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

"""Script to run a federated client on the heart-disease example."""

import argparse
import os
import sys

import numpy as np
import pandas as pd  # type: ignore
from declearn.communication import NetworkClientConfig
from declearn.dataset import InMemoryDataset
from declearn.main import FederatedClient

FILEDIR = os.path.dirname(os.path.abspath(__file__))
# Perform local imports.
sys.path.append(FILEDIR)
from data import get_data  # pylint: disable=wrong-import-order


def run_client(
    name: str,
    ca_cert: str,
) -> None:
    """Instantiate and run a given client.

    Arguments
    ---------
    name: str
        Name of the client (i.e. center data from which to use).
    ca_cert: str
        Path to the certificate authority file that was used to
        sign the server's SSL certificate.
    """

    # (1-2) Interface training and optional validation data.

    # Load and randomly split the dataset.
    path = os.path.join(FILEDIR, f"data/{name}.csv")
    if not os.path.isfile(path):
        get_data(os.path.join(FILEDIR, "data"), [name])
    data = pd.read_csv(path)
    data = data.loc[np.random.permutation(data.index)]
    n_tr = round(len(data) * 0.8)  # 80% train, 20% valid

    # Wrap train and validation data as Dataset objects.
    train = InMemoryDataset(
        data=data.iloc[:n_tr],
        target="num",
        expose_classes=True,  # share unique target labels with server
    )
    valid = InMemoryDataset(
        data=data.iloc[n_tr:],
        target="num",
    )

    # (3) Define network communication parameters.

    # Here, use websockets protocol on localhost:8765, with SSL encryption.
    network = NetworkClientConfig(
        protocol="websockets",
        server_uri="wss://localhost:8765",
        name=name,
        certificate=ca_cert,
    )

    # (4) Run any necessary import statement.
    #  => None are required in this example.

    # (5) Instantiate a FederatedClient and run it.

    client = FederatedClient(
        # fmt: off
        network, train, valid, checkpoint=f"{FILEDIR}/results/{name}"
        # Note: you may add `share_metrics=False` to prevent sending
        # evaluation metrics to the server, out of privacy concerns
    )
    client.run()


# Called when the script is called directly (using `python client.py`).
if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        type=str,
        help="name of your client",
        choices=["cleveland", "hungarian", "switzerland", "va"],
    )
    parser.add_argument(
        "--cert_path",
        dest="cert_path",
        type=str,
        help="path to the client-side ssl certification",
        default=os.path.join(FILEDIR, "ca-cert.pem"),
    )
    args = parser.parse_args()
    # Run the client routine.
    run_client(args.name, args.cert_path)
