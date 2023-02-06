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

"""Communication endpoints generic instantiation utils."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union


from declearn.communication.api import NetworkClient, NetworkServer
from declearn.utils import (
    TomlConfig,
    access_registered,
    access_types_mapping,
    dataclass_from_func,
)


__all__ = [
    "NetworkClientConfig",
    "NetworkServerConfig",
    "build_client",
    "build_server",
    "list_available_protocols",
    "_INSTALLABLE_BACKENDS",
]


_INSTALLABLE_BACKENDS = {}  # type: Dict[str, Tuple[str, ...]]


def raise_if_installable(
    protocol: str,
    exc: Optional[Exception] = None,
) -> None:
    """Raise a RuntimeError if a given protocol is missing but installable."""
    if protocol in _INSTALLABLE_BACKENDS:
        raise RuntimeError(
            f"The '{protocol}' communication protocol network endpoints "
            "could not be imported, but could be installed by satisfying "
            f"the following dependencies: {_INSTALLABLE_BACKENDS[protocol]}, "
            f"or by running `pip install declearn[{protocol}]`."
        ) from exc


def build_client(
    protocol: str,
    server_uri: str,
    name: str,
    certificate: Optional[str] = None,
    logger: Union[logging.Logger, str, None] = None,
    **kwargs: Any,
) -> NetworkClient:
    """Set up and return a NetworkClient communication endpoint.

    Parameters
    ----------
    protocol: str
        Name of the communications protocol backend, based on which
        the Client subclass to instantiate will be retrieved.
    server_uri: str
        Public uri of the server to which this client is to connect.
    name: str
        Name of this client, reported to the server for logging and
        messages' addressing purposes.
    certificate: str or None, default=None,
        Path to a certificate (publickey) PEM file, to use SSL/TLS
        communcations encryption.
    logger: logging.Logger or str or None, default=None,
        Logger to use, or name of a logger to set up using
        `declearn.utils.get_logger`. If None, use `type(client)-name`.
    **kwargs:
        Any valid additional keyword parameter may be passed as well.
        Refer to the target `NetworkClient` subclass for details.
    """
    protocol = protocol.strip().lower()
    try:
        cls = access_registered(name=protocol, group="NetworkClient")
    except KeyError as exc:
        raise_if_installable(protocol, exc)
        raise KeyError(
            "Failed to retrieve NetworkClient "
            f"class for protocol '{protocol}'."
        ) from exc
    assert issubclass(cls, NetworkClient)  # guaranteed by TypesRegistry
    return cls(server_uri, name, certificate, logger, **kwargs)


def build_server(
    protocol: str,
    host: str,
    port: int,
    certificate: Optional[str] = None,
    private_key: Optional[str] = None,
    password: Optional[str] = None,
    logger: Union[logging.Logger, str, None] = None,
    **kwargs: Any,
) -> NetworkServer:
    """Set up and return a NetworkServer communication endpoint.

    Parameters
    ----------
    protocol: str
        Name of the communications protocol backend, based on which
        the Server subclass to instantiate will be retrieved.
    host: str
        Host name (e.g. IP address) of the server.
    port: int
        Communications port to use.
    certificate: str or None, default=None
        Path to the server certificate (publickey) to use SSL/TLS
        communications encryption. If provided, `private_key` must
        be set as well.
    private_key: str or None, default=None
        Path to the server private key to use SSL/TLS communications
        encryption. If provided, `certificate` must be set as well.
    password: str or None, default=None
        Optional password used to access `private_key`, or path to a
        file from which to read such a password.
        If None but a password is needed, an input will be prompted.
    logger: logging.Logger or str or None, default=None,
        Logger to use, or name of a logger to set up with
        `declearn.utils.get_logger`. If None, use `type(server)`.
    **kwargs:
        Any valid additional keyword parameter may be passed as well.
        Refer to the target `NetworkServer` subclass for details.
    """
    # inherited signature; pylint: disable=too-many-arguments
    protocol = protocol.strip().lower()
    try:
        cls = access_registered(name=protocol, group="NetworkServer")
    except KeyError as exc:
        raise_if_installable(protocol, exc)
        raise KeyError(
            "Failed to retrieve NetworkServer "
            f"class for protocol '{protocol}'."
        ) from exc
    assert issubclass(cls, NetworkServer)  # guaranteed by TypesRegistry
    return cls(
        host, port, certificate, private_key, password, logger, **kwargs
    )


BuildClientConfig = dataclass_from_func(build_client)


BuildServerConfig = dataclass_from_func(build_server)


class NetworkClientConfig(BuildClientConfig, TomlConfig):  # type: ignore
    """TOML-parsable dataclass for network clients' instantiation."""

    def build_client(self) -> NetworkClient:
        """Build a NetworkClient from the wrapped parameters."""
        return self.call()


class NetworkServerConfig(BuildServerConfig, TomlConfig):  # type: ignore
    """TOML-parsable dataclass for network servers' instantiation."""

    def build_server(self) -> NetworkServer:
        """Build a NetworkServer from the wrapped parameters."""
        return self.call()


def list_available_protocols() -> List[str]:
    """Return the list of available network protocols.

    List protocol names that are associated with both a registered
    NetworkClient child class and a registered NetworkServer one.

    Note that registered implementations might include third-party ones
    thanks to the (automated) type-registration system attached to the
    base classes.

    Returns
    -------
    protocols: list[str]
        List of valid names that may be passed as 'protocol' so as
        to instantiate network endpoints through a generic builder
        such as the `build_client` or `build_server` exposed under
        `declearn.communication`.
    """
    client = access_types_mapping("NetworkClient")
    server = access_types_mapping("NetworkServer")
    return list(set(client).intersection(server))
