 # coding: utf-8

"""Base API to define client- and server-side communication endpoints.

This module provides the abstract base classes `Client` and `Server`,
that are used as communication endpoints by the `FederatedClient` and
`FederatedServer` classes.

It also exposes `flags` commonly used during communications.

Finally, it implements the `build_client` and `build_server` functions
that take advantage of types-registration to enable instantiating from
communication protocols' name - with the possibility to extend support
to third-party Client/Server subclasses via a simple class decorator
(see `declearn.utils.register_type`).
"""

from . import flags
from ._client import Client
from ._server import Server
from ._build import build_client, build_server
