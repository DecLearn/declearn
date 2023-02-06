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

"""Shared fixtures for declearn.communication module testing."""

import tempfile
from typing import Dict, Iterator

import pytest

from declearn.test_utils import generate_ssl_certificates


@pytest.fixture(name="ssl_cert", scope="module")
def ssl_cert_fixture() -> Iterator[Dict[str, str]]:
    """Fixture providing path to temporary self-signed TLS/SSL certificates.

    Return a dict containing paths to PEM files for a client (CA)
    certificate, a server certificate and a server private key,
    without any password protection.

    These files are located in a temporary directory that is deleted
    once this fixture goes out of scope. This fixture should be made
    to have a large scope to avoid unrequired repeated OpenSSL calls
    between tests.
    """
    with tempfile.TemporaryDirectory() as folder:
        ca_cert, sv_cert, sv_priv = generate_ssl_certificates(folder)
        yield {
            "client_cert": ca_cert,
            "server_cert": sv_cert,
            "server_pkey": sv_priv,
        }
