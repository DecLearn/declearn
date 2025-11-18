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

"""Submodule implementing client/server communications.

This is done by  defining server-side and client-side network communication
endpoints for federated learning processes, as well as suitable messages to
be transmitted, and the available communication protocols.

This module contains the following core submodules:

* [api][declearn.communication.api]:
    Base API to define client- and server-side communication endpoints.
* [utils][declearn.communication.utils]:
    Utils related to network communication endpoints' setup and usage.

It re-exports publicly from `utils` the following elements:

* [build_client][declearn.communication.build_client]:
    Instantiate a NetworkClient, selecting its subclass based on protocol name.
* [build_server][declearn.communication.build_server]:
    Instantiate a NetworkServer, selecting its subclass based on protocol name.
* [list_available_protocols][declearn.communication.list_available_protocols]:
    List the protocol names for which both a NetworkClient and NetworkServer
    classes are registered (hence available to `build_client`/`build_server`).
* [NetworkClientConfig][declearn.communication.NetworkClientConfig]:
    TOML-parsable dataclass for network clients' instantiation.
* [NetworkServerConfig][declearn.communication.NetworkServerConfig]:
    TOML-parsable dataclass for network servers' instantiation.

Finally, it defines the following protocol-specific submodules, provided
the associated third-party dependencies are available:

* [grpc][declearn.communication.grpc]:
    gRPC-based network communication endpoints.
    Requires the `grpcio` and `protobuf` third-party packages.
* [websockets][declearn.communication.websockets]:
    WebSockets-based network communication endpoints.
    Requires the `websockets` third-party package.
"""

# Messaging API and base tools:
from . import api, utils
from .utils import (
    _INSTALLABLE_BACKENDS,
    NetworkClientConfig,
    NetworkServerConfig,
    build_client,
    build_server,
    list_available_protocols,
)

# Concrete implementations using various protocols:
try:
    from . import grpc
except ImportError:  # pragma: no cover
    _INSTALLABLE_BACKENDS["grpc"] = ("grpcio", "protobuf")
try:
    from . import websockets
except ImportError:  # pragma: no cover
    _INSTALLABLE_BACKENDS["websockets"] = ("websockets",)
