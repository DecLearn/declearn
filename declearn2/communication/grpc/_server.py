# coding: utf-8

"""Server-side communication endpoint implementation using gRPC."""

import getpass
import os
from concurrent import futures
from typing import Optional

import grpc  # type: ignore
from cryptography.hazmat.primitives import serialization

from declearn2.communication.api import Server
from declearn2.communication.api._service import MessagesHandler
from declearn2.communication.grpc.protobufs import message_pb2
from declearn2.communication.grpc.protobufs.message_pb2_grpc import (
    MessageBoardServicer, add_MessageBoardServicer_to_server
)
from declearn2.utils import get_logger, register_type


def load_pem_file(
        path: str,
        password: Optional[str] = None
    ) -> bytes:
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


@register_type(name="grpc", group="Server")
class GrpcServer(Server):
    """Server-side communication endpoint using gRPC."""

    logger = get_logger("grpc-server")

    def __init__(
            self,
            host: str = 'localhost',
            port: int = 8765,
            certificate: Optional[str] = None,
            private_key: Optional[str] = None,
            password: Optional[str] = None,
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
        """
        # inherited signature; pylint: disable=too-many-arguments
        # Assign attributes and set up the gRPC server.
        super().__init__(host, port, certificate, private_key, password)
        self._server = None  # type: Optional[grpc.Server]

    @property
    def uri(self) -> str:
        return f"{self.host}:{self.port}"

    @staticmethod
    def _setup_ssl_context(
            certificate: Optional[str] = None,
            private_key: Optional[str] = None,
            password: Optional[str] = None,
        ) -> Optional[grpc.ServerCredentials]:
        """Set up and return an (optional) grpc.ServerCredentials object."""
        if (certificate is None) and (private_key is None):
            return None
        if (certificate is None) or (private_key is None):
            raise ValueError(
                "Both 'certificate' and 'private_key' are required "
                "to set up SSL encryption."
            )
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
        await self._server.start()

    def _setup_server(
            self,
        ) -> grpc.Server:
        """Set up and return a grpc Server to be used by this service."""
        server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        address = f'{self.host}:{self.port}'
        self.port = (
            server.add_secure_port(address, self._ssl)
            if (self._ssl is not None)
            else server.add_insecure_port(address)
        )
        servicer = GrpcServicer(self.handler)
        add_MessageBoardServicer_to_server(servicer, server)  # type: ignore
        return server

    async def stop(
            self,
        ) -> None:
        """Stop the gRPC server and purge information about clients."""
        if self._server is not None:
            await self._server.stop(grace=None)
            self._server = None
        await self.handler.purge()


class GrpcServicer(MessageBoardServicer):
    """A gRPC MessageBoard service to be used by a GrpcServer."""

    def __init__(
            self,
            handler: MessagesHandler,
        ) -> None:
        self.handler = handler

    async def ping(
            self,
            request: message_pb2.Empty,
            context: grpc.ServicerContext,
        ) -> message_pb2.Empty:
        """Handle a ping request from a client."""
        # async is needed; pylint: disable=invalid-overridden-method
        return message_pb2.Empty()

    async def send(
            self,
            request: message_pb2.Message,
            context: grpc.ServicerContext,
        ) -> message_pb2.Message:
        """Handle a Message-sending request from a client."""
        # async is needed; pylint: disable=invalid-overridden-method
        reply = await self.handler.handle_message(
            string=request.message,
            context=context.peer(),
        )
        return message_pb2.Message(message=reply.to_string())
