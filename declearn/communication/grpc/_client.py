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

"""Client-side communication endpoint implementation using gRPC"""

import logging
from typing import Any, ClassVar, Dict, Optional, Union

import grpc  # type: ignore

from declearn.communication.api import NetworkClient
from declearn.communication.grpc.protobufs import message_pb2
from declearn.communication.grpc.protobufs.message_pb2_grpc import (
    MessageBoardStub,
)

__all__ = [
    "GrpcClient",
]


CHUNK_LENGTH = 2**22 - 50  # 2**22 - sys.getsizeof("") - 1


class GrpcClient(NetworkClient):
    """Client-side communication endpoint using gRPC."""

    protocol: ClassVar[str] = "grpc"

    def __init__(
        self,
        server_uri: str,
        name: str,
        certificate: Optional[str] = None,
        logger: Union[logging.Logger, str, None] = None,
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
        logger: logging.Logger or str or None, default=None,
            Logger to use, or name of a logger to set up using
            `declearn.utils.get_logger`. If None, use `type(self)-name`.
        """
        super().__init__(server_uri, name, certificate, logger)
        self._channel: Optional[grpc.Channel] = None
        self._service: Optional[MessageBoardStub] = None

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

    async def start(self) -> None:
        if self._channel is None:
            self._channel = (
                grpc.aio.secure_channel(self.server_uri, self._ssl)  # type: ignore
                if (self._ssl is not None)
                else grpc.aio.insecure_channel(self.server_uri)
            )
        self._service = MessageBoardStub(self._channel)

    async def stop(self) -> None:
        if self._channel is not None:
            await self._channel.close()  # type: ignore
            self._channel = None
            self._service = None

    async def _send_message(
        self,
        message: str,
    ) -> str:
        """Send a message to the server and return the obtained reply."""
        if self._service is None:
            raise RuntimeError("Cannot send messages while not connected.")
        # Send the message, as a unary or as a stream of message chunks.
        if len(message) <= CHUNK_LENGTH:
            message = message_pb2.Message(message=message)
            replies = self._service.send(message)
        else:
            chunks = (
                message_pb2.Message(message=message[idx : idx + CHUNK_LENGTH])
                for idx in range(0, len(message), CHUNK_LENGTH)
            )
            replies = self._service.send_stream(chunks)
        # Collect the reply from a stream of message chunks.
        buffer = ""
        async for chunk in replies:
            buffer += chunk.message
        return buffer

    async def register(
        self,
        data_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            return await super().register(data_info)
        except grpc.aio.AioRpcError as err:
            self.logger.error(
                "Connection failed during registration: %s %s",
                err.code(),
                err.details(),
            )
            return False
