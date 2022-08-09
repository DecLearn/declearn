# coding: utf-8

"""Client-side communication endpoint implementation using gRPC"""

import asyncio
import json
from typing import Any, Dict, Optional, Tuple

import grpc  # type: ignore
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn2.communication.api import Client, flags
from declearn2.communication.grpc.protobufs.message_pb2 import (
    CheckMessageRequest, JoinRequest, Message,
)
from declearn2.communication.grpc.protobufs.message_pb2_grpc import (
    MessageBoardStub
)
from declearn2.utils import get_logger, json_pack, json_unpack


class GrpcClient(Client):
    """Client-side communication endpoint using gRPC."""

    logger = get_logger("websockets-client")

    def __init__(
            self,
            server_uri: str,
            name: str,
            certificate: Optional[str] = None,
            loop: Optional[asyncio.AbstractEventLoop] = None,
        ) -> None:
        """Instantiate the client-side gRPC communications handler.

        Parameters
        ----------
        server_uri: str
            Public uri of the gRPC server to which this client is to
            connect (e.g. "127.0.0.1:8765").
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
        # Assign attributes and handle TLS/SSL credentials.
        super().__init__(server_uri, name, loop=loop)
        self.credentials = self._setup_ssl_credentials(certificate)
        # Declare private attributes to store gRPC message-board servicers.
        self._channel = None  # type: Optional[grpc.Channel]
        self._service = None  # type: Optional[MessageBoardStub]

    @staticmethod
    def _setup_ssl_credentials(
            certificate: Optional[str] = None,
        ) -> Optional[grpc.ChannelCredentials]:
        """Set up and return an (optional) grpc ChannelCredentials object."""
        if certificate is None:
            return None
        with open(certificate, mode="rb") as file:
            cert_bytes = file.read()
        return grpc.ssl_channel_credentials(cert_bytes)

    def start(
            self
        ) -> None:
        self._channel = (
            grpc.aio.secure_channel(self.server_uri, self.credentials)
            if (self.credentials is not None)
            else grpc.aio.insecure_channel(self.server_uri)
        )
        self._service = MessageBoardStub(self._channel)  # type: ignore

    def stop(
            self
        ) -> None:
        if self._channel is not None:
            self.loop.run_until_complete(self._channel.close())
            self._channel = None
            self._service = None

    async def register(
            self,
            data_info: Dict[str, Any],
        ) -> Literal[  # type: ignore
            flags.FLAG_WELCOME, flags.FLAG_REFUSE_CONNECTION
        ]:
        if self._service is None:
            raise RuntimeError("Cannot register while not connected.")
        # Set up and send the join request.
        message = json.dumps(data_info, default=json_pack)
        request = JoinRequest(name=self.name, info=message)
        response = await self._service.join(request)
        reply = response.message
        if reply not in (flags.FLAG_WELCOME, flags.FLAG_REFUSE_CONNECTION):
            raise ValueError(
                f"Invalid server reply to registration request: '{reply}'."
            )
        return reply

    async def send_message(
            self,
            message: Dict[str, Any],
        ) -> None:
        if self._service is None:
            raise RuntimeError("Cannot send messages while not connected.")
        params = json.dumps(message, default=json_pack)
        message = Message(name=self.name, params=params)
        await self._service.send_message(message)

    async def check_message(
            self,
        ) -> Tuple[str, Dict[str, Any]]:
        if self._service is None:
            raise RuntimeError("Cannot receive messages while not connected.")
        # Request and await a message for this client.
        request = CheckMessageRequest(name=self.name)
        message = await self._service.check_message(request)
        # Unpack and deserialize the message, then return.
        action = message.action
        params = json.loads(message.params, object_hook=json_unpack)
        return action, params
