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

"""Script to run a federated client on the HuggingFace IMDb dataset example."""

import datetime
import logging
import os

import fire
import pandas as pd
from transformers import DistilBertTokenizer

import declearn

# Do not remove the following "unused" import,
# it is necessary for type registration
import declearn.model.torch
from declearn.dataset.torch import TorchDataset
from declearn.test_utils import make_importable

# Perform local imports.
with make_importable(os.path.dirname(__file__)):
    from dataset import TextDataset


FILEDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CERT = os.path.join(FILEDIR, "ca-cert.pem")


def run_client(
    client_name: str,
    data_folder: str,
    ca_cert: str = DEFAULT_CERT,
    protocol: str = "websockets",
    serv_uri: str = "wss://localhost:8765",
    verbose: bool = True,
) -> None:
    """Instantiate and run a given client.

    Parameters
    ---------
    client_name: str
        Name of the client (i.e. center data from which to use).
    data_folder: str
        The parent folder of this client's data
    ca_cert: str, default="./ca-cert.pem"
        Path to the certificate authority file that was used to
        sign the server's SSL certificate.
    protocol: str, default="websockets"
        Name of the communication protocol to use.
    serv_uri: str, default="wss://localhost:8765"
        URI of the server to which to connect.
    verbose: bool, default=True
        Whether to log everything to the console, or filter out most non-error
        information.
    """

    ### Optional: some convenience settings

    # Set GPU as prefered device
    declearn.utils.set_device_policy(gpu=True)

    # Set up logger and checkpointer
    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    checkpoint = os.path.join(FILEDIR, f"result_{stamp}", client_name)
    logger = declearn.utils.get_logger(
        name=client_name,
        fpath=os.path.join(checkpoint, "logs.txt"),
    )

    # Reduce logger verbosity
    if not verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(declearn.utils.LOGGING_LEVEL_MAJOR)

    ### (1-2) Interface training and optional validation data.

    # Target the proper dataset.
    data_folder = os.path.join(FILEDIR, data_folder, client_name)

    # Interface the data through the `TorchDataset` class.
    train_text_data = pd.read_csv(os.path.join(data_folder, "train_data.csv"))
    valid_text_data = pd.read_csv(os.path.join(data_folder, "valid_data.csv"))

    # Retrieve HuggingFace tokenizer matching the model.
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = TextDataset(train_text_data, tokenizer)
    valid_dataset = TextDataset(valid_text_data, tokenizer)

    train = TorchDataset(train_dataset)
    valid = TorchDataset(valid_dataset)

    ### (3) Define network communication parameters.

    # Here, by default, use websockets protocol on localhost:8765,
    # with SSL encryption.
    network = declearn.communication.build_client(
        protocol=protocol,
        server_uri=serv_uri,
        name=client_name,
        certificate=ca_cert,
    )

    ### (4) Instantiate a FederatedClient and run it.

    client = declearn.main.FederatedClient(
        netwk=network,
        train_data=train,
        valid_data=valid,
        checkpoint=checkpoint,
        logger=logger,
        verbose=verbose,
    )
    client.run()


# This part should not be altered: it provides with an argument parser
# for `python client.py`.


def main():
    "Fire-wrapped `run_client`."
    fire.Fire(run_client)


if __name__ == "__main__":
    main()
