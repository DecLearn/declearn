# coding: utf-8

"""Abstract class defining an API for client-side communication endpoints."""

import asyncio
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn2.communication.api.flags import (
    FLAG_WELCOME, FLAG_REFUSE_CONNECTION
)

__all__ = [
    'Client',
]

class Client(metaclass=ABCMeta):
    """Abstract class defining an API for client-side communication endpoints.

    This class defines the key methods used to communicate between a
    client and the orchestrating server during a federated learning
    process, agnostic to the actual communication protocol in use.
    """

    logger: logging.Logger

    @abstractmethod
    def __init__(
            self,
            server_uri: str,
            name: str,
            certificate: Optional[str] = None,
            loop: Optional[asyncio.AbstractEventLoop] = None,
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
        loop: asyncio.AbstractEventLoop or None, default=None
            An asyncio event loop to use.
            If None, use `asyncio.get_event_loop()`.
        """
        # Assign basic attributes. Note: children must handle 'certificate'.
        self.server_uri = server_uri
        self.name = name
        self.loop = asyncio.get_event_loop() if loop is None else loop

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
        try:
            self.loop.run_until_complete(task())
        finally:
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
    async def register(
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
            from `declearn.communication.api.flags`:
            - FLAG_WELCOME if the client was registered as participant
            - FLAG_REFUSE_CONNECTION if the registration was denied
        """
        return NotImplemented

    @abstractmethod
    async def send_message(
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
    async def check_message(
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
