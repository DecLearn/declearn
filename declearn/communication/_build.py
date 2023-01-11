# coding: utf-8

"""Communication endpoints generic instantiation utils."""

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple, Union


from declearn.communication.api import NetworkClient, NetworkServer
from declearn.utils import access_registered, access_types_mapping


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


@dataclasses.dataclass
class NetworkClientConfig:
    """Dataclass to store the configuration of a NetowkClient.

    Attributes
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
    kwargs: dict[str, None]
        Any valid additional keyword parameter may be passed as well.
        Refer to the target `NetworkClient` subclass for details.

    Notes
    -----
    This dataclass interfaces `declearn.communication.build_client`.
    Refer to it (and to the `declearn.communication` submodule) for
    additional details.
    """

    protocol: str
    server_uri: str
    name: str
    certificate: Optional[str] = None
    logger: Union[logging.Logger, str, None] = None
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this dataclass to a JSON-serializable dictionary."""
        return dataclasses.asdict(self)

    def build_client(self) -> NetworkClient:
        """Instantiate a NetworkClient based on this config."""
        params = self.to_dict()
        kwargs = params.pop("kwargs", {})
        return build_client(**params, **kwargs)


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


@dataclasses.dataclass
class NetworkServerConfig:
    """Dataclass to store the configuration of a communication Client.

    Attributes
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
    kwargs: dict[str, None]
        Any valid additional keyword parameter may be passed as well.
        Refer to the target `NetworkServer` subclass for details.

    Notes
    -----
    This dataclass interfaces `declearn.communication.build_server`.
    Refer to it (and to the `declearn.communication` submodule) for
    additional details.
    """

    # inherited signature; pylint: disable=too-many-instance-attributes

    protocol: str
    host: str
    port: int
    certificate: Optional[str] = None
    private_key: Optional[str] = None
    password: Optional[str] = None
    logger: Union[logging.Logger, str, None] = None
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this dataclass to a JSON-serializable dictionary."""
        return dataclasses.asdict(self)

    def build_server(self) -> NetworkServer:
        """Instantiate a NetworkServer based on this config."""
        params = self.to_dict()
        kwargs = params.pop("kwargs", {})
        return build_server(**params, **kwargs)


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
