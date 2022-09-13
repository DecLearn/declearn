# coding: utf-8

"""Abstract class defining an API for server-side communication endpoints."""

import asyncio
import logging
import types
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Set, Type


from declearn2.communication.api._service import MessagesHandler
from declearn2.communication.messaging import Message


__all__ = [
    'Server',
]


class Server(metaclass=ABCMeta):
    """Abstract class defining an API for server-side communication endpoints.

    This class defines the key methods used to communicate between
    a server and its clients during a federated learning process,
    agnostic to the actual communication protocol in use.

    Instantiating a `Server` does not instantly serve the declearn
    messaging program on the selected host and port. To enable clients
    to connect to the server via a `Server` object, its `start` method
    must first be awaited, and conversely, its `stop` method should be
    awaited to close the connection:
    >>> server = ServerSubclass(
    ...     "example.domain.com", 8765, "cert_path", "pkey_path"
    ... )
    >>> await server.start()
    >>> try:
    >>>     data_info = server.wait_for_clients(...)
    >>>     ...
    >>> finally:
    >>>     await server.stop()

    An alternative syntax to achieve the former is using the server
    object as an asynchronous context manager:
    >>> async with ServerSubclass(...) as server:
    >>>     data_info = server.wait_for_clients(...)
    >>>     ...

    Note that a `Server` manages an allow-list of clients, which is
    defined based on `Client.register(...)`-emitted requests during
    a registration phase restricted to the context of the awaitable
    `wait_for_clients` method.
    """

    logger: logging.Logger

    @abstractmethod
    def __init__(
            self,
            host: str,
            port: int,
            certificate: Optional[str] = None,
            private_key: Optional[str] = None,
            password: Optional[str] = None,
        ) -> None:
        """Instantiate the server-side communications handler.

        Parameters
        ----------
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
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        self.host = host
        self.port = port
        self._ssl = self._setup_ssl_context(certificate, private_key, password)
        self.handler = MessagesHandler(self.logger)

    @property
    @abstractmethod
    def uri(self) -> str:
        """URI on which this server is exposed, to be requested by clients."""
        return NotImplemented

    @property
    def client_names(self) -> Set[str]:
        """Set of registered clients' names."""
        return self.handler.client_names

    @staticmethod
    @abstractmethod
    def _setup_ssl_context(
            certificate: Optional[str] = None,
            private_key: Optional[str] = None,
            password: Optional[str] = None,
        ) -> Any:
        """Set up and return an (optional) SSL context object.

        The return type is communication-protocol dependent.
        """
        return NotImplemented

    @abstractmethod
    async def start(
            self,
        ) -> None:
        """Initialize the server and start welcoming communications."""
        return None

    @abstractmethod
    async def stop(
            self,
        ) -> None:
        """Stop the server and purge information about clients."""
        return None

    async def __aenter__(
            self,
        ) -> 'Server':
        await self.start()
        return self

    async def __aexit__(
            self,
            exc_type: Type[Exception],
            exc_value: Exception,
            exc_tb: types.TracebackType,
        ) -> None:
        await self.stop()

    async def wait_for_clients(
            self,
            min_clients: int = 1,
            max_clients: Optional[int] = None,
            timeout: Optional[int] = None,
        ) -> Dict[str, Dict[str, Any]]:
        """Wait for clients to register for training, with given criteria.

        Parameters
        ----------
        min_clients: int, default=1
            Minimum number of clients required. Corrected to be >= 1.
            If `timeout` is None, used as the exact number of clients
            required - once reached, registration will be closed.
        max_clients: int or None, default=None
            Maximum number of clients authorized to register.
        timeout: int or None, default=None
            Optional maximum waiting time (in seconds) beyond which
            to close registration and either return or raise.

        Raises
        ------
        RuntimeError:
            If the number of registered clients does not abide by the
            provided boundaries at the end of the process.


        Returns
        -------
        client_info: dict[str, dict[str, any]]
            A dictionary where the keys are the participants
            and the values are their information.
        """
        return await self.handler.wait_for_clients(
            min_clients, max_clients, timeout
        )

    async def broadcast_message(
            self,
            message: Message,
            clients: Optional[Set[str]] = None,
            heartbeat: int = 1,
            timeout: Optional[int] = None,
        ) -> None:
        """Send a message to an ensemble of clients and await its collection.

        Parameters
        ----------
        message: Message
            Message instance that is to be delivered to the client.
        clients: set[str] or None, default=None
            Optional subset of registered clients, messages from
            whom to wait for. If None, set to `self.client_names`.
        heartbeat: int, default=1
            Delay (in seconds) between verifications that the message
            has been collected by the client.
        timeout: int or None, default=None
            Optional maximum delay (in seconds) beyond which to stop
            waiting for collection and raise an asyncio.TimeoutError.

        Raises
        ------
        asyncio.TimeoutError:
            If `timeout` is set and is reached while the message is
            yet to be collected by at least one of the clients.
        """
        if clients is None:
            clients = self.client_names
        messages = {client: message for client in clients}
        await self.send_messages(messages, heartbeat, timeout)

    async def send_messages(
            self,
            messages: Dict[str, Message],
            heartbeat: int = 1,
            timeout: Optional[int] = None,
        ) -> None:
        """Send a message to an ensemble of clients and await its collection.

        Parameters
        ----------
        messages: dict[str, Message]
            Dict mapping Message instances that are to be delivered
            to the names of their destinatory client.
        heartbeat: int, default=1
            Delay (in seconds) between verifications that the message
            has been collected by the client.
        timeout: int or None, default=None
            Optional maximum delay (in seconds) beyond which to stop
            waiting for collection and raise an asyncio.TimeoutError.

        Raises
        ------
        asyncio.TimeoutError:
            If `timeout` is set and is reached while the message is
            yet to be collected by at least one of the clients.
        """
        routines = [
            self.send_message(message, client, heartbeat, timeout)
            for client, message in messages.items()
        ]
        results = await asyncio.gather(*routines, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                raise result

    async def send_message(
            self,
            message: Message,
            client: str,
            heartbeat: int = 1,
            timeout: Optional[int] = None,
        ) -> None:
        """Send a message to a given client and wait for it to be collected.

        Parameters
        ----------
        message: Message
            Message instance that is to be delivered to the client.
        client: str
            Identifier of the client to whom the message is addressed.
        heartbeat: int, default=1
            Delay (in seconds) between verifications that the message
            has been collected by the client.
        timeout: int or None, default=None
            Optional maximum delay (in seconds) beyond which to stop
            waiting for collection and raise an asyncio.TimeoutError.

        Raises
        ------
        asyncio.TimeoutError:
            If `timeout` is set and is reached while the message is
            yet to be collected by the client.
        """
        await self.handler.send_message(message, client, heartbeat, timeout)

    async def wait_for_messages(
            self,
            clients: Optional[Set[str]] = None,
            heartbeat: int = 1,
            timeout: Optional[int] = None,
        ) -> Dict[str, Message]:
        """Wait for an ensemble of clients to have sent a message.

        Parameters
        ----------
        clients: set[str] or None, default=None
            Optional subset of registered clients, messages from
            whom to wait for. If None, set to `self.client_names`.
        hearbeat: int, default=1
            Delay (in seconds) between verifications that a client
            has sent their message.
        timeout: int or None, default=None
            Optional maximum delay (in seconds) beyond which to stop
            waiting for messages and raise an asyncio.TimeoutError.

        Raises
        ------
        asyncio.TimeoutError:
            If any of the clients has failed to deliver a message
            before `timeout` was reached.

        Returns
        -------
        messages: dict[str, Message]
            A dictionary where the keys are the clients' names and
            the values are Message objects they sent to the server.
        """
        if clients is None:
            clients = self.client_names
        routines = [
            self.handler.recv_message(client, heartbeat, timeout)
            for client in clients
        ]
        received = await asyncio.gather(*routines, return_exceptions=True)
        messages = {}  # type: Dict[str, Message]
        for client, message in zip(clients, received):
            if isinstance(message, Exception):
                raise message
            messages[client] = message
        return messages
