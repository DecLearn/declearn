# coding: utf-8

"""Abstract class defining an API for client-side communication endpoints."""

import asyncio
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Coroutine, Dict, Optional


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

    async def register(
            self,
            data_info: Dict[str, Any],
        ) -> bool:
        """Request the server to join a federating learning session

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
