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
    Instantiate a Client, selecting its subclass based on protocol name.
* build_server:
    Instantiate a Server, selecting its subclass based on protocol name.

Note: the latter two functions natively support the declearn-implemented
      network protocols listed above, but will be extended to any third-
      party implementation of Client and Server subclasses, provided the
      `declearn.utils.register_type(name=protocol_name, group="Client")`
      decorator is used (with adequate protocol_name string and "Server"
      group for Server subclasses).
"""

# Messaging and Communications API and base tools:
from . import messaging
from . import api
from ._build import build_client, build_server
# Concrete implementations using various protocols:
from . import grpc
from . import websockets
