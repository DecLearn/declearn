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

"""Utils to set up command-line argument parsers for declearn examples."""

import argparse
from typing import Optional

__all__ = [
    "setup_client_argparse",
    "setup_server_argparse",
]


def setup_client_argparse(
    usage: Optional[str] = None,
    default_uri: str = "wss://localhost:8765",
    default_ptcl: str = "websockets",
    default_cert: str = "./ca-cert.pem",
) -> argparse.ArgumentParser:
    """Set up an ArgumentParser to be used in a client-side script.

    Arguments
    ---------
    usage: str or None, default=None
        Optional usage string to add to the ArgumentParser.
    default_uri: str, default="wss://localhost:8765"
        Default value for the 'uri' argument.
    default_ptcl: str, default="websockets"
        Default value for the 'protocol' argument.
    default_cert: str, default="./ca-cert.pem"
        Default value for the 'certificate' argument.

    Returns
    -------
    parser: argparse.ArgumentParser
        ArgumentParser with pre-set optional arguments required
        to configure network communications on the client side.
    """
    parser = argparse.ArgumentParser(
        usage=usage,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--uri",
        dest="uri",
        type=str,
        help="server URI to which to connect",
        default=default_uri,
    )
    parser.add_argument(
        "--protocol",
        dest="protocol",
        type=str,
        help="name of the communication protocol to use",
        default=default_ptcl,
    )
    parser.add_argument(
        "--cert",
        dest="certificate",
        type=str,
        help="path to the client-side ssl certificate authority file",
        default=default_cert,
    )
    return parser


# pylint: disable-next=too-many-positional-arguments
def setup_server_argparse(  # noqa: PLR0913
    usage: Optional[str] = None,
    default_host: str = "localhost",
    default_port: int = 8765,
    default_ptcl: str = "websockets",
    default_cert: str = "./server-cert.pem",
    default_pkey: str = "./server-pkey.pem",
) -> argparse.ArgumentParser:
    """Set up an ArgumentParser to be used in a server-side script.

    Arguments
    ---------
    usage: str or None, default=None
        Optional usage string to add to the ArgumentParser.
    default_host: str, default="localhost"
        Default value for the 'host' argument.
    default_port: int, default=8765
        Default value for the 'port' argument.
    default_ptcl: str, default="websockets"
        Default value for the 'protocol' argument.
    default_cert: str, default="./server-cert.pem"
        Default value for the 'certificate' argument.
    default_pkey: str, default="./server-pkey.pem"
        Default value for the 'private_key' argument.

    Returns
    -------
    parser: argparse.ArgumentParser
        ArgumentParser with pre-set optional arguments required
        to configure network communications on the server side.
    """
    # arguments serve modularity; pylint: disable=too-many-arguments
    parser = argparse.ArgumentParser(
        usage=usage,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        dest="host",
        type=str,
        help="hostname or IP address on which to serve",
        default=default_host,
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        help="communication port on which to serve",
        default=default_port,
    )
    parser.add_argument(
        "--protocol",
        dest="protocol",
        type=str,
        help="name of the communication protocol to use",
        default=default_ptcl,
    )
    parser.add_argument(
        "--cert",
        dest="certificate",
        type=str,
        help="path to the server-side ssl certificate",
        default=default_cert,
    )
    parser.add_argument(
        "--pkey",
        dest="private_key",
        type=str,
        help="path to the server-side ssl private key",
        default=default_pkey,
    )
    return parser
