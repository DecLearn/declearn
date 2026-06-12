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

import os
from dataclasses import astuple, dataclass
from typing import Union

from dataset import MaskedAutoEncoderDataset
from sklearn.model_selection import train_test_split

from declearn.communication.utils._build import NetworkClientConfig
from declearn.dataset.examples import EMGDatasetConfigs, load_semg_hand_poses
from declearn.dataset.torch import TorchDataset
from declearn.main._client import FederatedClient
from declearn.test_utils._argparse import setup_client_argparse

FILEDIR = os.path.dirname(__file__)


@dataclass
class ClientConfigInput:
    name : Union[str| list[str]]
    certificate : str
    folder: str = "time-series/data"
    protocol : str = "websockets"
    server_uri : str = "wss://localhost:8765"
    verbose : bool = True
    

def run_client(
    configs : ClientConfigInput
):
    "Creates and runs a client instance"

    params = EMGDatasetConfigs(folder=configs.folder)
    semg_data = load_semg_hand_poses(params)
    train, valid= train_test_split(semg_data, test_size=0.20)
    
    train = TorchDataset(
        dataset=MaskedAutoEncoderDataset(train),
        
    )
    valid = TorchDataset(
        dataset= MaskedAutoEncoderDataset(valid),
    )

    name, certificate, _, protocol, server_uri, _ = astuple(configs)
    network = NetworkClientConfig(
        protocol, 
        server_uri, 
        name, 
        certificate
    )

    client = FederatedClient(
        netwk=network, 
        train_data=train,
        valid_data=valid, 
        
    )

    client.run()


if __name__== "__main__":
    #parse any neccessary arguments from bash using the client parser
    parser = setup_client_argparse(
        usage="Start a client providing an EMG dataset", 
        default_cert=os.path.join(FILEDIR, "ca-cert.pem"),
    )
    parser.add_argument("--folder", type=str, help="Folder in which the time-series data is located.")
    parser.add_argument("--name", type=str, help="Name of the client")
    parser.add_argument("--window_size", type=int, help="Sliding window size")
    parser.add_argument("--target", type=int, help="Target column to extract", choices=list(range(1,9)))
    
    args= parser.parse_args() 

    #set up the configs object
    client_configs = ClientConfigInput(
        name=args.name,
        folder=args.folder,
        certificate=args.certificate,
        protocol=args.protocol,
        server_uri=args.uri)
    
    # run the client routine
    run_client(client_configs)
    
