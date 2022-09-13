# coding: utf-8

"""Client-side communication endpoint implementation using gRPC"""

from typing import Any, Dict, Optional

import grpc  # type: ignore

from declearn2.communication.api import Client
from declearn2.communication.messaging import (
    Message, parse_message_from_string
)
from declearn2.communication.grpc.protobufs import message_pb2
from declearn2.communication.grpc.protobufs.message_pb2_grpc import (
    MessageBoardStub
)
from declearn2.utils import get_logger, register_type


@register_type(name="grpc", group="Client")
class GrpcClient(Client):
    """Client-side communication endpoint using gRPC."""

    logger = get_logger("grpc-client")

    def __init__(
            self,
            server_uri: str,
            name: str,
            certificate: Optional[str] = None,
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
        """
        super().__init__(server_uri, name, certificate)
        self._channel = None  # type: Optional[grpc.Channel]
        self._service = None  # type: Optional[MessageBoardStub]

    @staticmethod
    def _setup_ssl_context(
            certificate: Optional[str] = None,
        ) -> Optional[grpc.ChannelCredentials]:
        """Set up and return an (optional) grpc ChannelCredentials object."""
        if certificate is None:
            return None
        with open(certificate, mode="rb") as file:
            cert_bytes = file.read()
        return grpc.ssl_channel_credentials(cert_bytes)

    async def start(
            self
        ) -> None:
        if self._channel is None:
            self._channel = (
                grpc.aio.secure_channel(self.server_uri, self._ssl)
                if (self._ssl is not None)
                else grpc.aio.insecure_channel(self.server_uri)
            )
        self._service = MessageBoardStub(self._channel)  # type: ignore

    async def stop(
            self
        ) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._service = None

    async def _send_message(
            self,
            message: Message,
        ) -> Message:
        """Send a message to the server and return the obtained reply."""
        if self._service is None:
            raise RuntimeError("Cannot send messages while not connected.")
        grpc_message = message_pb2.Message(message=message.to_string())
        grpc_reply = await self._service.send(grpc_message)
        return parse_message_from_string(grpc_reply.message)

    async def register(
            self,
            data_info: Dict[str, Any],
        ) -> bool:
        try:
            return await super().register(data_info)
        except grpc.aio.AioRpcError as err:
            self.logger.error(
                "Connection failed during registration: %s %s",
                err.code(), err.details()
            )
            return False
