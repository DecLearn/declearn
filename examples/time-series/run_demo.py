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

"""Demonstration script using the Hand Poses sEMG dataset."""

import os
import tempfile

import fire

from declearn.utils import make_importable, run_as_processes, setup_root_logger
from declearn.utils.examples import generate_ssl_certificates

# Perform local imports.
with make_importable(os.path.dirname(__file__)):
    from model import ModelConfigsInput
    from prepare_data import prepare_data_for_clients
    from run_client import ClientConfigInput, run_client
    from run_server import ServerConfigInput, run_server


def run_demo(
    nb_clients: int = 2,
    window_size: int = 128,
    mask_ratio: float = 0.25,
):
    """Runs 1 server simulation along max 2 clients demo
    for time-series example

    Parameters
    ----------
    nb_clients: int (optional)
        Number of clients for the experiment. Defaults to 2.
    window_size: int (optional)
        Size of the sliding window applied on every sEMG signal.
        Defaults to 128.
    mask_ratio: float (optional)
        The rate of random points to be masked. Defaults to 0.25.

    Raises
    ------
    RuntimeError:
        - If the processes do not yield successful response then
            there must be an error during runtime.
    """
    # Setup Declearn root logger (to display Declearn logs on stderr).
    setup_root_logger()

    with tempfile.TemporaryDirectory() as tempdir:
        prepare_data_for_clients(nb_clients)

        ca_cert, sv_cert, sv_pkey = generate_ssl_certificates(tempdir)
        server_configs = ServerConfigInput(
            nb_clients=nb_clients,
            certificate=sv_cert,
            private_key=sv_pkey,
        )
        model_configs = ModelConfigsInput(
            input_dim=window_size, mask_ratio=mask_ratio
        )

        server = (run_server, (server_configs, model_configs))

        clients = [
            (
                run_client,
                (
                    ClientConfigInput(
                        name=f"client_{i}",
                        certificate=ca_cert,
                        data_path=os.path.join(
                            os.path.join(os.path.dirname(__file__), "data"),
                            f"client_{i}.pt",
                        ),
                        target=i,
                    ),
                ),
            )
            for i in range(nb_clients)
        ]
        success, outp = run_as_processes(server, *clients)

        if not success:
            raise RuntimeError(
                "Something went wrong during the demo. Exceptions caught:\n"
                "\n".join(str(e) for e in outp if isinstance(e, RuntimeError))
            )


if __name__ == "__main__":
    fire.Fire(run_demo)
