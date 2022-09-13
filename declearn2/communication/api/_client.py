# coding: utf-8

"""Abstract class defining an API for client-side communication endpoints."""

import logging
import types
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Type


from declearn2.communication.messaging import (
    Empty, Error, GetMessageRequest, JoinReply, JoinRequest, Message
)


__all__ = [
    'Client',
]

class Client(metaclass=ABCMeta):
    """Abstract class defining an API for client-side communication endpoints.

    This class defines the key methods used to communicate between a
    client and the orchestrating server during a federated learning
    process, agnostic to the actual communication protocol in use.

    Instantiating a `Client` does not trigger a connection to the
    target server. To enable communicating with the server via a
    `Client` object, its `start` method must first be awaited and
    conversely, its `stop` method should be awaited to close the
    connection:
    >>> client = ClientSubclass("example.domain.com:8765", "name", "cert_path")
    >>> await client.start()
    >>> try:
    >>>     client.register(data_info)
    >>>     ...
    >>> finally:
    >>>     await client.stop()

    An alternative syntax to achieve the former is using the client
    object as an asynchronous context manager:
    >>> async with ClientSubclass(...) as client:
    >>>     client.register(data_info)
    >>>     ...

    Note that a declearn `Server` manages an allow-list of clients,
    which is defined during a registration phase of limited time,
    based on requests emitted through the `Client.register` method.
    Any message emitted using `Client.send_message` will probably
    be rejected by the server if the client has not been registered.
    """

    logger: logging.Logger

    @abstractmethod
    def __init__(
            self,
            server_uri: str,
            name: str,
            certificate: Optional[str] = None,
        ) -> None:
        """Instantiate the client-side communications handler.

        Parameters
        ----------
        server_uri: str
            Public uri of the server to which this client is to connect.
        name: str
            Name of this client, reported to the server for logging and
            messages' addressing purposes.
        certificate: str or None, default=None,
            Path to a certificate (publickey) PEM file, to use SSL/TLS
            communcations encryption.
        """
        # Assign basic attributes. Note: children must handle 'certificate'.
        self.server_uri = server_uri
        self.name = name
        self._ssl = self._setup_ssl_context(certificate)

    @staticmethod
    @abstractmethod
    def _setup_ssl_context(
            certificate: Optional[str] = None,
        ) -> Any:
        """Set up and return an (optional) SSL context object.

        The return type is communication-protocol dependent.
        """
        return NotImplemented

    @abstractmethod
    async def start(
            self
        ) -> None:
        """Start the client, i.e. connect to the server.

        Note: this method can be called safely even if the
        client is already running (simply having no effect).
        """
        return None

    @abstractmethod
    async def stop(
            self
        ) -> None:
        """Stop the client, i.e. close all connections.

        Note: this method can be called safely even if the
        client is not running (simply having no effect).
        """
        return None

    async def __aenter__(
            self,
        ) -> 'Client':
        await self.start()
        return self

    async def __aexit__(
            self,
            exc_type: Type[Exception],
            exc_value: Exception,
            exc_tb: types.TracebackType,
        ) -> None:
        await self.stop()

    async def register(
            self,
            data_info: Dict[str, Any],
        ) -> bool:
        """Request the server to join a federating learning session.

        Parameters
        ----------
        data_info : dict[str, any]
            JSON-serializable dictionary holding information on the local
            data that the server will use to set up the training model.

        Returns
        -------
        accepted: bool
            Whether the registration request was accepted by the server
            or not.

        Raises
        -------
        TypeError:
            If the server does not return a JoinReply message.
        """
        reply = await self._send_message(JoinRequest(self.name, data_info))
        # Case when a JoinReply was received.
        if isinstance(reply, JoinReply):
            self.logger.info(
                "Registration was %saccepted: '%s'",
                "" if reply.accept else "not ", reply.flag
            )
            return reply.accept
        # Case when an Error was received.
        if isinstance(reply, Error):
            self.logger.error(
                "Registration request triggered an error:\n%s", reply.message
            )
            return False
        # Otherwise, raise.
        raise TypeError(
            "Received an undue message type in response to JoinRequest."
        )

    @abstractmethod
    async def _send_message(
            self,
            message: Message,
        ) -> Message:
        """Send a message to the server and return the obtained reply.

        This method should be defined by concrete Client subclasses,
        and implement communication-protocol-specific code to send a
        Message (of any kind) to the server and await the primary
        reply from the `MessagesHandler` used by the server.
        """
        return NotImplemented

    async def send_message(
            self,
            message: Message,
        ) -> None:
        """Send a message to the server.

        Parameters
        ----------
        message: Message
            Message instance that is to be delivered to the server.

        Raises
        ------
        RuntimeError:
            If the server emits an Error message in response to the
            message sent.
        TypeError:
            If the server returns a non-Empty message.

        Note
        ----
        The message sent here is designed to be received using the
        `Server.wait_for_messages` method.
        """
        reply = await self._send_message(message)
        if isinstance(reply, Empty):
            return None
        if isinstance(reply, Error):
            raise RuntimeError(
                f"Message was rejected with error: {reply.message}"
            )
        raise TypeError(
            "Received an undue message type in response to the posted message."
        )

    async def check_message(
            self,
            timeout: Optional[int] = None
        ) -> Message:
        """Retrieve the next message sent by the server.

        Returns
        -------
        action: str
            Instruction for the client.
        params: dict
            Associated parameters, as a JSON-serializable dict.

        Note
        ----
        The message received here is expected to have been sent
        using one of the following `Server` methods: `send_message`,
        `send_messages`, or `broadcast_message`.
        """
        return await self._send_message(GetMessageRequest(timeout))
