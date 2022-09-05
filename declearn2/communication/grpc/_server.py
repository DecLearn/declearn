# coding: utf-8

"""Server-side communication endpoint implementation using gRPC."""

import asyncio
import getpass
import os
from concurrent import futures
from typing import Optional

import grpc  # type: ignore
from cryptography.hazmat.primitives import serialization

from declearn2.communication.api import Server
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
            port: int = 0,
            certificate: Optional[str] = None,
            private_key: Optional[str] = None,
            password: Optional[str] = None,
            loop: Optional[asyncio.AbstractEventLoop] = None,
        ) -> None:
        """Instantiate the server-side gRPC communications handler.

        Parameters
        ----------
        host : str, default='localhost'
            Host name (e.g. IP address) of the server.
        port: int, default=0
            Communications port to use.
            If left to zero, gRPC runtime will choose one.
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
        # inherited signature; pylint: disable=too-many-arguments
        # Assign attributes and set up the gRPC server.
        super().__init__(host, port, loop=loop)
        self._server = self._setup_server(certificate, private_key, password)

    @property
    def uri(self) -> str:
        return f"{self.host}:{self.port}"

    def _setup_server(
            self,
            certificate: Optional[str] = None,
            private_key: Optional[str] = None,
            password: Optional[str] = None,
        ) -> grpc.Server:
        """Set up and return a grpc Server to be used by this service."""
        server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        creds = self._setup_ssl_credentials(certificate, private_key, password)
        address = f'{self.host}:{self.port}'
        self.port = (
            server.add_secure_port(address, creds)
            if (creds is not None)
            else server.add_insecure_port(address)
        )
        servicer = self._setup_board_servicer()
        add_MessageBoardServicer_to_server(servicer, server)  # type: ignore
        return server

    @staticmethod
    def _setup_ssl_credentials(
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

    def _setup_board_servicer(
            self,
        ) -> MessageBoardServicer:
        """Create and return a gRPC message board servicer for this server."""
        server = self
        # Declare a MessageBoardServicer implementing message processing.
        class Servicer(MessageBoardServicer):
            """A gRPC MessageBoard service to be used by this GrpcServer."""

            async def ping(
                    self,
                    request: message_pb2.Empty,
                    context: grpc.ServicerContext,
                ) -> message_pb2.Empty:
                """Handle a ping request from a client."""
                return message_pb2.Empty()

            async def send(
                    self,
                    request: message_pb2.Message,
                    context: grpc.ServicerContext,
                ) -> message_pb2.Message:
                """Handle a Message-sending request from a client."""
                # async is needed; pylint: disable=invalid-overridden-method
                nonlocal server
                reply = await server.handler.handle_message(
                    string=request.message,
                    context=context.peer(),
                )
                return message_pb2.Message(message=reply.to_string())
        # Instantiate and return the servicer.
        return Servicer()

    def start(
            self,
        ) -> None:
        """Start the gRPC server."""
        self.loop.run_until_complete(self._server.start())

    def stop(
            self,
        ) -> None:
        """Stop the gRPC server."""
        self.loop.run_until_complete(self._server.stop(grace=None))
