# coding: utf-8

"""Submodule implementing client/server communications.

This module contains the following core submodules:
* api:
    Base API to define client- and server-side communication endpoints.
* messaging:
    Message dataclasses defining information containers to be exchanged
    between communication endpoints.


It also exposes the following core utility functions:
* build_client:
    Instantiate a NetworkClient, selecting its subclass based on protocol name.
* build_server:
    Instantiate a NetworkServer, selecting its subclass based on protocol name.
* list_available_protocols:
    List the protocol names for which both a NetworkClient and NetworkServer
    classes are registered (hence available to `build_client`/`build_server`).


Finally, it defines the following protocol-specific submodules, provided
the associated third-party dependencies are available:
* grpc:
    gRPC-based network communication endpoints.
    Requires the `grpcio` and `protobuf` third-party packages.
* websockets:
    WebSockets-based network communication endpoints.
    Requires the `websockets` third-party package.
"""

# Messaging and Communications API and base tools:
from . import messaging
from . import api
from ._build import (
    NetworkClientConfig,
    NetworkServerConfig,
    build_client,
    build_server,
    list_available_protocols,
    _INSTALLABLE_BACKENDS,
)

# Concrete implementations using various protocols:
try:
    from . import grpc
except ImportError:
    _INSTALLABLE_BACKENDS["grpc"] = ("grpcio", "protobuf")
try:
    from . import websockets
except ImportError:
    _INSTALLABLE_BACKENDS["websockets"] = ("websockets",)
