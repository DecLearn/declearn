# coding: utf-8

"""Communication endpoints generic instantiation utils."""

import dataclasses
from typing import Any, Dict, Optional


from declearn2.communication.api import Client, Server
from declearn2.utils import access_registered, create_types_registry


__all__ = [
    'NetworkClientConfig',
    'NetworkServerConfig',
    'build_client',
    'build_server',
]


# Create a pair of type registries.
create_types_registry("Client", base=Client)
create_types_registry("Server", base=Server)


def build_client(
        protocol: str,
        server_uri: str,
        name: str,
        certificate: Optional[str] = None,
        **kwargs: Any,
    ) -> Client:
    """Set up and return a Client communication endpoint.

    Note: this function requires the target Client subclass to have
          been registered with name `protocol` under the 'Client'
          registry group. See `declearn.utils.register_type`.

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
    **kwargs:
        Any valid additional keyword parameter may be passed as well.
        Refer to the target `Client` subclass for details.
    """
    try:
        cls = access_registered(name=protocol.lower(), group="Client")
    except KeyError as exc:
        raise KeyError(
            f"Failed to retrieve Client class for protocol '{protocol}'."
        ) from exc
    assert issubclass(cls, Client)  # guaranteed by TypesRegistry
    return cls(server_uri, name, certificate, **kwargs)


@dataclasses.dataclass
class NetworkClientConfig:
    """Dataclass to store the configuration of a communication Client.

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
    kwargs: dict[str, None]
        Any valid additional keyword parameter may be passed as well.
        Refer to the target `Client` subclass for details.

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
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this dataclass to a JSON-serializable dictionary."""
        return dataclasses.asdict(self)

    def build_client(self) -> Client:
        """Instantiate a communication Client based on this config."""
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
        **kwargs: Any,
    ) -> Server:
    """Set up and return a Server communication endpoint.

    Note: this function requires the target Server subclass to have
          been registered with name `protocol` under the 'Server'
          registry group. See `declearn.utils.register_type`.

    Parameters
    ----------
    protocol: str
        Name of the communications protocol backend, based on which
        the Server subclass to instantiate will be retrieved.
    host : str
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
    **kwargs:
        Any valid additional keyword parameter may be passed as well.
        Refer to the target `Server` subclass for details.
    """
    # inherited signature; pylint: disable=too-many-arguments
    try:
        cls = access_registered(name=protocol.lower(), group="Server")
    except KeyError as exc:
        raise KeyError(
            f"Failed to retrieve Server class for protocol '{protocol}'."
        ) from exc
    assert issubclass(cls, Server)  # guaranteed by TypesRegistry
    return cls(host, port, certificate, private_key, password, **kwargs)


@dataclasses.dataclass
class NetworkServerConfig:
    """Dataclass to store the configuration of a communication Client.

    Attributes
    ----------
    protocol: str
        Name of the communications protocol backend, based on which
        the Server subclass to instantiate will be retrieved.
    host : str
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
    kwargs: dict[str, None]
        Any valid additional keyword parameter may be passed as well.
        Refer to the target `Server` subclass for details.

    Notes
    -----
    This dataclass interfaces `declearn.communication.build_server`.
    Refer to it (and to the `declearn.communication` submodule) for
    additional details.
    """

    protocol: str
    host: str
    port: int
    certificate: Optional[str] = None
    private_key: Optional[str] = None
    password: Optional[str] = None
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this dataclass to a JSON-serializable dictionary."""
        return dataclasses.asdict(self)

    def build_server(self) -> Server:
        """Instantiate a communication Server based on this config."""
        params = self.to_dict()
        kwargs = params.pop("kwargs", {})
        return build_server(**params, **kwargs)
