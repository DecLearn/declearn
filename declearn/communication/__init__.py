# coding: utf-8

"""Submodule implementing client/server communications.

This module contains the following submodules:
* api:
    Base API to define client- and server-side communication endpoints.
* messaging:
    Message dataclasses defining information containers to be exchanged
    between communication endpoints.
* grpc:
    gRPC-based network communication endpoints.
* websockets:
    WebSockets-based network communication endpoints.

It also exposes the following functions:
* build_client:
    Instantiate a NetworkClient, selecting its subclass based on protocol name.
* build_server:
    Instantiate a NetworkServer, selecting its subclass based on protocol name.

Note: the latter two functions natively support the declearn-implemented
      network protocols listed above, but will be extended to any third-
      party implementation of NetworkClient and NetworkServer subclasses.
"""

# Messaging and Communications API and base tools:
from . import messaging
from . import api
from ._build import (
    NetworkClientConfig,
    NetworkServerConfig,
    build_client,
    build_server,
)

# Concrete implementations using various protocols:
from . import grpc
from . import websockets
