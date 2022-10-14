"""Demonstration script using the UCI Heart Disease Dataset."""

import os
import sys
import tempfile

# Perform local imports.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from client import run_client  # pylint: disable=wrong-import-position
from server import run_server  # pylint: disable=wrong-import-position

from declearn.test_utils import (
    generate_ssl_certificates,
    run_as_processes,
)


NAMES = ["cleveland", "hungarian", "switzerland", "va"]


def run_demo(
    nb_clients: int = 4,
) -> None:
    """Run a server and its clients using multiprocessing."""
    # Use a temporary directory for single-use self-signed SSL files.
    with tempfile.TemporaryDirectory() as folder:
        # Generate self-signed SSL certificates and gather their paths.
        ca_cert, sv_cert, sv_pkey = generate_ssl_certificates(folder)
        # Specify the server and client routines that need executing.
        server = (run_server, (nb_clients, sv_cert, sv_pkey))
        clients = [
            (run_client, (name, ca_cert)) for name in NAMES[:nb_clients]
        ]
        # Run routines in isolated processes. Raise if any failed.
        exitcodes = run_as_processes(server, *clients)
        if any(code != 0 for code in exitcodes):
            raise RuntimeError("Something went wrong during the demo.")


if __name__ == "__main__":
    run_demo()
