# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Server-side communication endpoint implementation using gRPC."""

import getpass
import logging
import os
import warnings
from concurrent import futures
from typing import AsyncIterator, Optional, Union

import grpc  # type: ignore
from cryptography.hazmat.primitives import serialization

from declearn.communication.api import NetworkServer
from declearn.communication.api.backend import MessagesHandler, actions, flags
from declearn.communication.grpc.protobufs import message_pb2
from declearn.communication.grpc.protobufs.message_pb2_grpc import (
    MessageBoardServicer,
    add_MessageBoardServicer_to_server,
)
from declearn.communication.utils._compression import resolve_grpc_compression

__all__ = [
    "GrpcServer",
]


CHUNK_LENGTH = 2**22 - 16  # max_size - protobuf overhead with a safety margin


def load_pem_file(path: str, password: Optional[str] = None) -> bytes:
    """Load the content of a PEM file."""
    # Load the raw bytes data from the PEM file.
    with open(path, mode="rb") as file:
        pem_bytes = file.read()
    # If a password is required and missing, prompt for one.
    if ("ENCRYPTED".encode() in pem_bytes[:20]) and not password:
        password = getpass.getpass("Enter PEM pass phrase:")
    # Optionally decode the data using the provided password.
    if password:
        if os.path.isfile(password):
            with open(password, mode="r", encoding="utf-8") as file:
                password = file.read().strip("\n")
        pwd_bytes = password.encode()
        private_key = serialization.load_pem_private_key(pem_bytes, pwd_bytes)
        return private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    # Otherwise, return the raw bytes.
    return pem_bytes


class GrpcServer(NetworkServer):
    """Server-side communication endpoint using gRPC."""

    protocol = "grpc"

    handler: MessagesHandler

    # pylint: disable-next=too-many-positional-arguments
    # TODO for 2.10 : remove deprecated "logger" argument
    def __init__(  # noqa: PLR0913
        self,
        host: str = "localhost",
        port: int = 8765,
        certificate: Optional[str] = None,
        private_key: Optional[str] = None,
        password: Optional[str] = None,
        heartbeat: float = 1.0,
        compression: Optional[str] = None,
        logger: Union[logging.Logger, str, None] = None,
    ) -> None:
        """Instantiate the server-side gRPC communications handler.

        Parameters
        ----------
        host : str, default='localhost'
            Host name (e.g. IP address) of the server.
        port: int, default=8765
            Communications port to use.
            If set to 0, the gRPC runtime will choose one when the
            server is first started.
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
        heartbeat: float, default=1.0
            Delay (in seconds) between verifications when checking for a
            message having beend received from or collected by a client.
        compression: str or None, default=None,
            Optional message-level compression to use over the wire.
            One of ``None``, ``"none"`` (both = no compression) or
            ``"deflate"``. Invalid values raise ``ValueError`` on
            construction.
        logger: logging.Logger or str or None, default=None,
            Deprecated in v2.8, removed in v2.10.
            Not used anymore.
        """
        # inherited signature; pylint: disable=too-many-arguments
        if logger is not None:
            warnings.warn(
                "Argument 'logger' is deprecated and useless now, it will be "
                "removed in 2.10. "
                "To customize the instance logger, you may use instead "
                "logging utils from `declearn.utils` or the 'logging' Python "
                "module.",
                DeprecationWarning,
                stacklevel=2,
            )
        # Assign attributes and set up the gRPC server.
        super().__init__(
            host, port, certificate, private_key, password, heartbeat
        )
        self.compression = compression
        self._server: Optional[grpc.Server] = None

    @property
    def uri(self) -> str:
        return f"{self.host}:{self.port}"

    @staticmethod
    def _setup_ssl_context(
        certificate: str,
        private_key: str,
        password: Optional[str] = None,
    ) -> Optional[grpc.ServerCredentials]:
        """Set up and return a grpc.ServerCredentials object."""
        cert = load_pem_file(certificate)
        pkey = load_pem_file(private_key, password)
        return grpc.ssl_server_credentials(
            private_key_certificate_chain_pairs=[(pkey, cert)],
            root_certificates=None,
            require_client_auth=False,
        )

    async def start(
        self,
    ) -> None:
        """Start the gRPC server."""
        self._server = self._setup_server()
        self.logger.info("Server is now starting...")
        await self._server.start()  # type: ignore

    def _setup_server(
        self,
    ) -> grpc.Server:
        """Set up and return a grpc Server to be used by this service."""
        server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            compression=resolve_grpc_compression(self.compression),
        )
        address = f"{self.host}:{self.port}"
        self.port = (
            server.add_secure_port(address, self._ssl)
            if (self._ssl is not None)
            else server.add_insecure_port(address)
        )
        servicer = GrpcServicer(self.handler)
        add_MessageBoardServicer_to_server(servicer, server)
        return server  # type: ignore

    async def stop(
        self,
    ) -> None:
        """Stop the gRPC server and purge information about clients."""
        if self._server is not None:
            await self._server.stop(grace=None)  # type: ignore
            self._server = None
        await self.handler.purge()


class GrpcServicer(MessageBoardServicer):
    """A gRPC MessageBoard service to be used by a GrpcServer."""

    # NOTE: mypy complains about protobuf-generated classes, hence
    #       the amount of "type: ignore" comments in this code.

    def __init__(
        self,
        handler: MessagesHandler,
    ) -> None:
        self.handler = handler

    async def ping(
        self,
        request: message_pb2.Empty,  # type: ignore
        context: grpc.ServicerContext,
    ) -> message_pb2.Empty:  # type: ignore
        """Handle a ping request from a client."""
        # async is needed; pylint: disable=invalid-overridden-method
        return message_pb2.Empty()

    async def _handle_and_reply(
        self,
        bin_msg: bytes,
        context: grpc.ServicerContext,
    ) -> AsyncIterator[message_pb2.Message]:  # type: ignore
        """Handle a received message and send back the (chunked) reply."""
        reply = await self.handler.handle_message(bin_msg, context.peer())
        bin_reply = reply.serialize()
        for srt in range(0, len(bin_reply), CHUNK_LENGTH):
            end = srt + CHUNK_LENGTH
            yield message_pb2.Message(message=bin_reply[srt:end])

    async def send(
        self,
        request: message_pb2.Message,  # type: ignore
        context: grpc.ServicerContext,
    ) -> AsyncIterator[message_pb2.Message]:  # type: ignore
        """Handle a Message-sending request from a client."""
        # async is needed; pylint: disable=invalid-overridden-method
        bin_msg = request.message  # type: ignore
        async for chunk in self._handle_and_reply(bin_msg, context):
            yield chunk

    async def send_stream(
        self,
        request_iterator: AsyncIterator[message_pb2.Message],  # type: ignore
        context: grpc.ServicerContext,
    ) -> AsyncIterator[message_pb2.Message]:  # type: ignore
        """Handle a chunked Message-sending request from a client."""
        # async is needed; pylint: disable=invalid-overridden-method
        # Case when an unknown peer attempts sending a stream: send an error.
        if context.peer() not in self.handler.registered_clients:
            error = actions.Reject(flags.REJECT_UNREGISTERED_CHUNKED)
            self.handler.logger.warning(
                "Refused a chunks-streaming request from client %s",
                context.peer(),
            )
            yield message_pb2.Message(message=error.serialize())
        # Otherwise, assemble the message for streamed chunks, then reply.
        else:
            req_chunks = []
            async for request in request_iterator:
                req_chunks.append(request.message)
            bin_msg = b"".join(req_chunks)
            async for chunk in self._handle_and_reply(bin_msg, context):
                yield chunk
