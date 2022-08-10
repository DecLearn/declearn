# coding: utf-8

"""Abstract class defining an API for server-side communication endpoints."""

import asyncio
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Coroutine, Dict, Optional, Set

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
            nb_clients: int,  # revise: move this to "wait_for_clients"
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
        nb_clients: int
            Maximum number of clients that should be accepted.
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
        self.nb_clients = nb_clients
        self.host = host
        self.port = port
        self.loop = asyncio.get_event_loop() if loop is None else loop

    @property
    @abstractmethod
    def uri(self) -> str:
        """URI on which this server is exposed, to be requested by clients."""
        return NotImplemented

    @property
    @abstractmethod
    def client_names(self) -> Set[str]:
        """Set of registered clients' names."""
        return NotImplemented

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

    @abstractmethod
    async def wait_for_clients(
            self,
        ) -> Dict[str, Dict[str, Any]]:
        """Pause the server until the required number of clients have joined.

        Returns
        -------
        client_info: dict[str, dict[str, any]]
            A dictionary where the keys are the participants
            and the values are their information.
        """
        return NotImplemented

    @abstractmethod
    def broadcast_message(
            self,
            action: str,
            params: Dict[str, Any],
        ) -> None:
        """Send a message to all the clients.

        Parameters
        ----------
        action: str
            Instruction for the client.
        params: dict
            Associated parameters, as a JSON-serializable dict.

        Note
        ----
        The message sent here (action and params) is designed to
        be received using the `Client.wait_for_message` method.
        """
        return None

    @abstractmethod
    async def send_message(
            self,
            client: str,
            action: str,
            params: Dict[str, Any]
        ) -> None:
        """Send a message to a given client.

        Parameters
        ----------
        client: str
            Identifier of the client to whom the message is addressed.
        action: str
            Instruction for the client.
        params: dict
            Associated parameters, as a JSON-serializable dict.

        Note
        ----
        The message sent here (action and params) is designed to
        be received using the `Client.wait_for_message` method.
        """
        return None

    @abstractmethod
    async def wait_for_messages(
            self,
        ) -> Dict[str, Dict[str, Any]]:
        """Retrieve the next expected messages from each of the participants.

        Returns
        -------
        messages:
            A dictionary where the keys are the participants and
            the values are messages they sent to the server.

        Note
        ----
        The messages received here are expected to have been sent
        usin the `Client.send_message` method.
        """
        return NotImplemented
