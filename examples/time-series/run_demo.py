
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
import tempfile

import fire

from declearn.test_utils import generate_ssl_certificates, make_importable
from declearn.utils import run_as_processes

# Perform local imports.
with make_importable(os.path.dirname(__file__)):
    from model import ModelConfigsInput
    from run_client import ClientConfigInput, run_client
    from run_server import ServerConfigInput, run_server



def run_demo(folder : str, nb_clients : int = 2, window_size : int = 128, mask_ratio: float = 0.25): 
    """Runs 1 server simulation along max 2 clients demo for time-series example"""
    
    with tempfile.TemporaryDirectory() as tempdir: 
        ca_cert, sv_cert, sv_pkey = generate_ssl_certificates(tempdir)
        server_configs = ServerConfigInput(
            nb_clients, 
            certificate=sv_cert,
            private_key=sv_pkey,
        )
        model_configs = ModelConfigsInput(input_dim=window_size,mask_ratio=mask_ratio)
        server = (run_server, (server_configs, model_configs,))
        

        clients = [
            (run_client, (ClientConfigInput(f"client_{i}", ca_cert, folder),)) 
            for i in range(nb_clients)]
        success, outp = run_as_processes(server, *clients)
        
        if not success:
            raise RuntimeError(
                "Something went wrong during the demo. Exceptions caught:\n"
                "\n".join(str(e) for e in outp if isinstance(e, RuntimeError))
            )
            
if __name__ == "__main__":
    fire.Fire(run_demo)

