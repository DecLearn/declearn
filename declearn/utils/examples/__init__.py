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

"""Shared utilities used in DecLearn examples.

Argument-parsing utils
----------------------
* [setup_client_argparse][declearn.utils.examples.setup_client_argparse]:
    Set up an `ArgumentParser` to be used in a client-side script.
* [setup_server_argparse][declearn.utils.examples.setup_server_argparse]:
    Set up an `ArgumentParser` to be used in a server-side script.

SSL certificate utils
---------------------
* [generate_ssl_certificates]\
[declearn.utils.examples.generate_ssl_certificates]
    Generate a self-signed CA and a CA-signed SSL certificate.
"""

__all__ = [
    "setup_client_argparse",
    "setup_server_argparse",
    "generate_ssl_certificates",
]

from ._argparse import setup_client_argparse, setup_server_argparse
from ._gen_ssl import generate_ssl_certificates
