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

"""Demonstration script using the UCI Heart Disease Dataset."""

import os
import tempfile

import fire

from declearn.utils import make_importable, run_as_processes
from declearn.utils.examples import generate_ssl_certificates

# Perform local imports.
with make_importable(os.path.dirname(__file__)):
    from run_client import run_client
    from run_server import run_server

NAMES = ["cleveland", "hungarian", "switzerland", "va"]


def run_demo(
    nb_clients: int = 4,
) -> None:
    """Run a server and its clients using multiprocessing."""
    if not (1 <= nb_clients <= 4):
        raise ValueError(
            "This demo only supports 1 to 4 clients. \nReceived "
            f"{nb_clients}. Please use a valid input."
        )

    # Use a temporary directory for single-use self-signed SSL files.
    with tempfile.TemporaryDirectory() as folder:
        # Generate self-signed SSL certificates and gather their paths.
        ca_cert, sv_cert, sv_pkey = generate_ssl_certificates(folder)
        # Specify the server and client routines that need executing.
        server = (run_server, (nb_clients, sv_cert, sv_pkey))
        clients = [
            (run_client, {"name": name, "ca_cert": ca_cert, "verbose": False})
            for name in NAMES[:nb_clients]
        ]
        # Run routines in isolated processes. Raise if any failed.
        success, outp = run_as_processes(server, *clients)
        if not success:
            exceptions = "\n".join(
                str(e) for e in outp if isinstance(e, RuntimeError)
            )
            raise RuntimeError(
                "Something went wrong during the demo. Exceptions caught:\n"
                + exceptions
            )


if __name__ == "__main__":
    fire.Fire(run_demo)
