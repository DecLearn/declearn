# coding: utf-8

"""Base classes to define communication endpoints.

This module provides the abstract base classes `Client` and `Server`,
that are used as communication endpoints by the `FederatedClient` and
`FederatedServer` classes.
"""

import asyncio
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn2.communication.api.flags import (
    FLAG_WELCOME, FLAG_REFUSE_CONNECTION
)


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
        """Send a message to all the clients

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


class Client(metaclass=ABCMeta):
    """Abstract class defining an API for client-side communication endpoints.

    This class defines the key methods used to communicate between a
    client and the orchestrating server during a federated learning
    process, agnostic to the actual communication protocol in use.
    """

    def run_until_complete(
            self,
            task: Callable[[], Coroutine[Any, Any, None]],
        ) -> None:
        """Start a client to run a given task, and stop it afterwards.

        Parameters
        ----------
        task: callable returning an awaitable
            The coroutine function to perform, using this client.
        """
        self.start()
        loop = getattr(self, '_loop', asyncio.get_event_loop())  # revise
        loop.run_until_complete(task())
        self.stop()

    @abstractmethod  # revise
    def start(
            self
        ) -> None:
        """Start the client, i.e. connect to the server."""
        return None

    @abstractmethod  # revise
    def stop(
            self
        ) -> None:
        """Stop the client, i.e. close all connections."""
        return None

    @abstractmethod
    def register(
            self,
            data_info: Dict[str, Any],
        ) -> Literal[FLAG_WELCOME, FLAG_REFUSE_CONNECTION]:  # type: ignore
        """Request the server to join a federating learning session

        Parameters
        ----------
        data_info : dict[str, any]
            JSON-serializable dictionary holding information on the local
            data that the server will use to set up the training model.

        Returns
        -------
        response: str
            The return code to the registration request, using a flag
            from `declearn.communication.messages`:
            - FLAG_WELCOME if the client was registered as participant
            - FLAG_REFUSE_CONNECTION if the registration was denied
        """
        return NotImplemented

    @abstractmethod
    def send_message(
            self,
            message: Dict[str, Any],
        ) -> None:
        """Send a message to the server.

        Parameters
        ----------
        message: dict[str, Any]
            A JSON-serializable dictionary with str keys, holding
            the information being sent to the server.

        Note
        ----
        The message sent here is designed to be received using the
        `Server.wait_for_messages` method.
        """
        return None

    @abstractmethod
    def check_message(
            self,
        ) -> Tuple[str, Dict[str, Any]]:
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
        usin the `Server.broadcast_message` method.
        """
        return NotImplemented
