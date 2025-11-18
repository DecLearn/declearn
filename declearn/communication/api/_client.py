# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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

"""Abstract class defining an API for client-side communication endpoints."""

import abc
import asyncio
import logging
import types
import warnings
from typing import Any, ClassVar, Dict, Optional, Self, Type, Union

from declearn.communication.api.backend import flags

# Drop,  # FUTURE: implement a method to drop politely
from declearn.communication.api.backend.actions import (
    Accept,
    ActionMessage,
    Join,
    Ping,
    Recv,
    Reject,
    Send,
    parse_action_from_string,
)
from declearn.messaging import Message, SerializedMessage
from declearn.utils import create_types_registry, get_logger, register_type
from declearn.version import VERSION

__all__ = [
    "NetworkClient",
]


@create_types_registry
class NetworkClient(metaclass=abc.ABCMeta):
    """Abstract class defining an API for client-side communication endpoints.

    This class defines the key methods used to communicate between a
    client and the orchestrating server during a federated learning
    process, agnostic to the actual communication protocol in use.

    Instantiating a `NetworkClient` does not trigger a connection to
    the target server. To enable communicating with the server via a
    `NetworkClient` object, its `start` method must first be awaited
    and conversely, its `stop` method should be awaited to close the
    connection:
    ```
    >>> client = ClientSubclass("example.domain.com:8765", "name", "cert_path")
    >>> await client.start()
    >>> try:
    >>>     client.register()
    >>>     ...
    >>> finally:
    >>>     await client.stop()
    ```

    An alternative syntax to achieve the former is using the client
    object as an asynchronous context manager:
    ```
    >>> async with ClientSubclass(...) as client:
    >>>     client.register()
    >>>     ...
    ```

    Note that a declearn `NetworkServer` manages an allow-list of
    clients, which is defined during a registration phase of limited
    time, based on requests emitted through the `NetworkClient.register`
    method. Any message emitted using `NetworkClient.send_message` will
    probably be rejected by the server if the client has not registered.
    """

    protocol: ClassVar[str] = NotImplemented
    """Protocol name identifier, unique across NetworkClient classes."""

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automate the type-registration of NetworkClient subclasses."""
        super().__init_subclass__(**kwargs)
        if register:
            register_type(cls, cls.protocol, group="NetworkClient")

    def __init__(
        self,
        server_uri: str,
        name: str,
        certificate: Optional[str] = None,
        logger: Union[logging.Logger, str, None] = None,
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
        logger: logging.Logger or str or None, default=None,
            Logger to use, or name of a logger to set up using
            `declearn.utils.get_logger`. If None, use `type(self)-name`.
        """
        self.server_uri = server_uri
        self.name = name
        self._ssl = self._setup_ssl_context(certificate)
        if isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            self.logger = get_logger(logger or f"{type(self).__name__}-{name}")

    @staticmethod
    @abc.abstractmethod
    def _setup_ssl_context(
        certificate: Optional[str] = None,
    ) -> Any:
        """Set up and return an (optional) SSL context object.

        The return type is communication-protocol dependent.
        """

    # similar to NetworkServer API; pylint: disable=duplicate-code

    @abc.abstractmethod
    async def start(
        self,
    ) -> None:
        """Start the client, i.e. connect to the server.

        Note: this method can be called safely even if the
        client is already running (simply having no effect).
        """

    @abc.abstractmethod
    async def stop(
        self,
    ) -> None:
        """Stop the client, i.e. close all connections.

        Note: this method can be called safely even if the
        client is not running (simply having no effect).
        """

    async def __aenter__(
        self,
    ) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Type[Exception],
        exc_value: Exception,
        exc_tb: types.TracebackType,
    ) -> None:
        await self.stop()

    # pylint: enable=duplicate-code

    @abc.abstractmethod
    async def _send_message(
        self,
        message: str,
    ) -> str:
        """Send a message to the server and return the obtained reply.

        This method should be defined by concrete NetworkClient child
        classes, and implement communication-protocol-specific code
        to send a message to the server and await the primary reply
        from the `MessagesHandler` used by the server.
        """

    async def _exchange_action_messages(
        self,
        message: ActionMessage,
    ) -> ActionMessage:
        """Send an `ActionMessage` to the server and await its response."""
        query = message.to_string()
        reply = await self._send_message(query)
        try:
            return parse_action_from_string(reply)
        except Exception as exc:
            error = "Failed to decode a reply from the server."
            self.logger.critical(error)
            raise RuntimeError(error) from exc

    async def register(
        self,
        data_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register to the server as a client.

        Returns
        -------
        accepted: bool
            Whether the registration request was accepted by the server.

        Raises
        -------
        TypeError
            If the server does not return a valid message.
            This is a failsafe and should never happen.
        """
        if data_info:
            warnings.warn(
                "Sending dataset information is no longer part of the client "
                "registration process. The argument was ignored, and will be "
                "removed in DecLearn version 2.4 and/or 3.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        query = Join(name=self.name, version=VERSION)
        reply = await self._exchange_action_messages(query)
        # Case when registration was accepted.
        if isinstance(reply, Accept):
            self.logger.info("Registration was accepted: '%s'", reply.flag)
            return True
        # Case when registration was rejected.
        if isinstance(reply, Reject):
            self.logger.error("Registration was rejected: '%s'", reply.flag)
            return False
        # Otherwise, raise.
        error = (
            "Received an undue response type when attempting to register "
            f"with the server: '{type(reply)}'."
        )
        self.logger.critical(error)
        raise TypeError(error)

    async def send_message(
        self,
        message: Message,
    ) -> None:
        """Send a message to the server.

        Parameters
        ----------
        message: str
            Message instance that is to be delivered to the server.

        Raises
        ------
        RuntimeError
            If the server rejects the sent message.
        TypeError
            If the server returns neither a ping-back nor rejection message.
            This is a failsafe and should never happen.

        Note
        ----
        The message sent here is designed to be received using the
        `NetworkServer.wait_for_messages` method.
        """
        query = Send(message.to_string())
        reply = await self._exchange_action_messages(query)
        if isinstance(reply, Ping):
            return None
        if isinstance(reply, Reject):
            error = f"Message was rejected: {reply.flag}"
            self.logger.error(error)
            raise RuntimeError(error)
        error = (
            "Received an undue response type when attempting to send a "
            f"message: '{type(reply)}'."
        )
        self.logger.critical(error)
        raise TypeError(error)

    async def recv_message(
        self,
        timeout: Optional[float] = None,
    ) -> SerializedMessage:
        """Await a message from the server, with optional timeout.

        Parameters
        ----------
        timeout: float or None, default=None
            Optional timeout delay, after which the server will send
            a timeout notification to this client if no message is
            available for it.

        Returns
        -------
        message: SerializedMessage
            Serialized message received from the server.

        Note
        ----
        The message received here is expected to have been sent
        using one of the following `NetorkServer` methods:
        `send_message`, `send_messages`, or `broadcast_message`.

        Raises
        ------
        asyncio.TimeoutError
            If no message is available after the `timeout` delay.
        RuntimeError
            If the request is rejected by the server.
        TypeError
            If the server returns data of unproper type.
            This is a failsafe and should never happen.
        """
        # Send a query, get a reply and return its content when possible.
        query = Recv(timeout)
        reply = await self._exchange_action_messages(query)
        if isinstance(reply, Send):
            return SerializedMessage.from_message_string(reply.content)
        # Handle the various kinds of failures and raise accordingly.
        if isinstance(reply, Reject):
            if reply.flag == flags.CHECK_MESSAGE_TIMEOUT:
                error = "Message-retrieval request timed out."
                self.logger.error(error)
                raise asyncio.TimeoutError(error)
            error = f"Message-retrieval request was rejected: '{reply.flag}'."
            self.logger.error(error)
            raise RuntimeError(error)
        error = (
            "Received an undue response type when attempting to receive a "
            f"message: '{type(reply)}'."
        )
        self.logger.critical(error)
        raise TypeError(error)
