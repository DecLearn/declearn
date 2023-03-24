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

"""Base API to define client- and server-side communication endpoints.

This module provides `NetworkClient` and `NetworkServer`, two abstract
base classes that are to be used as network communication endpoints for
federated learning processes:

* [NetworkClient][declearn.communication.api.NetworkClient]:
    Abstract class defining an API for client-side communication endpoints.
* [NetworkServer][declearn.communication.api.NetworkServer]:
    Abstract class defining an API for server-side communication endpoints.
"""

from ._client import NetworkClient
from ._server import NetworkServer
