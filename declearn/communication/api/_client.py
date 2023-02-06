# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

import logging
import types
from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Type, Union

from declearn.communication.messaging import (
    Empty,
    Error,
    GetMessageRequest,
    JoinReply,
    JoinRequest,
    Message,
)
from declearn.utils import create_types_registry, get_logger, register_type

__all__ = [
    "NetworkClient",
]


@create_types_registry
class NetworkClient(metaclass=ABCMeta):
    """Abstract class defining an API for client-side communication endpoints.

    This class defines the key methods used to communicate between a
    client and the orchestrating server during a federated learning
    process, agnostic to the actual communication protocol in use.

    Instantiating a `NetworkClient` does not trigger a connection to
    the target server. To enable communicating with the server via a
    `NetworkClient` object, its `start` method must first be awaited
    and conversely, its `stop` method should be awaited to close the
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

    Note that a declearn `NetworkServer` manages an allow-list of
    clients, which is defined during a registration phase of limited
    time, based on requests emitted through the `NetworkClient.register`
    method. Any message emitted using `NetworkClient.send_message` will
    probably be rejected by the server if the client has not registered.
    """

    protocol: ClassVar[str] = NotImplemented

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automate the type-registration of NetworkClient subclasses."""
        super().__init_subclass__(**kwargs)
        if register:
            register_type(cls, cls.protocol, group="NetworkClient")

    @abstractmethod
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
        # Assign basic attributes. Note: children must handle 'certificate'.
        self.server_uri = server_uri
        self.name = name
        self._ssl = self._setup_ssl_context(certificate)
        if isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            self.logger = get_logger(logger or f"{type(self).__name__}-{name}")

    @staticmethod
    @abstractmethod
    def _setup_ssl_context(
        certificate: Optional[str] = None,
    ) -> Any:
        """Set up and return an (optional) SSL context object.

        The return type is communication-protocol dependent.
        """

    # similar to NetworkServer API; pylint: disable=duplicate-code

    @abstractmethod
    async def start(self) -> None:
        """Start the client, i.e. connect to the server.

        Note: this method can be called safely even if the
        client is already running (simply having no effect).
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop the client, i.e. close all connections.

        Note: this method can be called safely even if the
        client is not running (simply having no effect).
        """

    async def __aenter__(
        self,
    ) -> "NetworkClient":
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
                "" if reply.accept else "not ",
                reply.flag,
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

        This method should be defined by concrete NetworkClient child
        classes, and implement communication-protocol-specific code
        to send a Message (of any kind) to the server and await the
        primary reply from the `MessagesHandler` used by the server.
        """

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
        `NetworkServer.wait_for_messages` method.
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

    async def check_message(self, timeout: Optional[int] = None) -> Message:
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
        using one of the following `NetorkServer` methods:
        `send_message`, `send_messages`, or `broadcast_message`.
        """
        return await self._send_message(GetMessageRequest(timeout))
