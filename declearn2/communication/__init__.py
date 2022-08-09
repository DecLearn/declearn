# coding: utf-8

"""Submodule implementing client/server communications.

This module contains the following submodules:
* api:
    Base API to define client- and server-side communication endpoints.
* flags (re-exposed from `api`):
    Set of communication flags conventionally used in declearn.
* grpc:
    gRPC-based network communication endpoints.
* websockets:
    WebSockets-based network communication endpoints.

It also exposes the following functions (re-exported from `api`):
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

from . import api
from . import grpc
from . import websockets
# Re-expose some utils to limit import paths' length.
from .api import build_client, build_server, flags
