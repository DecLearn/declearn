# coding: utf-8

"""Abstract class defining an API for server-side communication endpoints."""

import asyncio
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Coroutine, Dict, Optional, Set


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
            loop: Optional[asyncio.AbstractEventLoop] = None,
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
        loop: asyncio.AbstractEventLoop or None, default=None
            An asyncio event loop to use.
            If None, use `asyncio.get_event_loop()`.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        self.host = host
        self.port = port
        self.loop = asyncio.get_event_loop() if loop is None else loop
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

    @abstractmethod
    def start(
            self,
        ) -> None:
        """Initialize the server and start welcoming communications."""
        return None

    @abstractmethod
    def stop(
            self,
        ) -> None:
        """Stop the server and purge information about clients."""
        return None

    def run_until_complete(
            self,
            task: Callable[[], Coroutine[Any, Any, None]],
        ) -> None:
        """Start a server to run a given task, and stop it afterwards.

        Parameters
        ----------
        task: callable returning a coroutine
            The coroutine function to perform, using this server.
        """
        self.start()
        try:
            self.loop.run_until_complete(task())
        except asyncio.CancelledError as cerr:
            msg = f"Asyncio error while running the server: {cerr}"
            self.logger.info(msg)
        finally:
            self.stop()

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
