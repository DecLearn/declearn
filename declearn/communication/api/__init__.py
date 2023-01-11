# coding: utf-8

"""Base API to define client- and server-side communication endpoints.

This module provides `NetworkClient` and `NetworkServer`, two abstract
base classes that are to be used as network communication endpoints for
federated learning processes.
"""

from ._client import NetworkClient
from ._server import NetworkServer
