# coding: utf-8

"""Server-side communication endpoint implementation using gRPC."""

import asyncio
import getpass
import json
import os
from typing import Any, Dict, Optional

import grpc  # type: ignore
from cryptography.hazmat.primitives import serialization

from declearn2.communication.api import Server
from declearn2.communication.grpc._service import Service
from declearn2.utils import get_logger, json_pack, register_type


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


register_type(name="grpc", group="Server")
class GrpcServer(Server):
    """Server-side communication endpoint using gRPC."""

    logger = get_logger("grpc-server")

    def __init__(
            self,
            nb_clients: int,
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
        nb_clients: int
            Maximum number of clients that should be accepted.
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
        # Assign attributes and set up optional TLS/SSL credentials.
        super().__init__(nb_clients, host, port, loop=loop)
        credentials = self._setup_ssl_credentials(
            certificate, private_key, password
        )
        # Instantiate a gRPC message-board servicer. Override host/port info.
        self._service = Service(nb_clients, host, port, credentials)
        self.host = self._service.host
        self.port = self._service.port

    @property
    def uri(self) -> str:
        return f"{self.host}:{self.port}"

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
            private_key_certificate_chain_pairs=(pkey, cert),
            root_certificates=None,
            require_client_auth=False,
        )

    def start(
            self,
        ) -> None:
        """Start the gRPC server."""
        self.loop.run_until_complete(self._service.start())

    def stop(
            self,
        ) -> None:
        """Stop the gRPC server."""
        self.loop.run_until_complete(self._service.stop())

    async def wait_for_clients(
            self,
        ) -> Dict[str, Dict[str, Any]]:
        self.logger.info("Waiting for clients to register for training...")
        number = -1
        while len(self._service.registered_users) < self.nb_clients:
            await asyncio.sleep(1)  # past: self.heartbeat
            if len(self._service.registered_users) != number:
                self.logger.info(
                    "Now waiting for %s additional participants.",
                    self.nb_clients - len(self._service.registered_users)
                )
                number = len(self._service.registered_users)
        return self._service.registered_users

    def broadcast_message(
            self,
            action: str,
            params: Dict[str, Any],
        ) -> None:
        # Serialize the parameters. Note: gRPC handles dicts as messages.
        dump = json.dumps(params, default=json_pack)
        message = {"action": action, "params": dump}
        # Set the message up for transmission.
        for client, c_info in self._service.registered_users.items():
            name = c_info["alias"]
            if client in self._service.outgoing_messages:
                self.logger.warning(
                    "Overwriting pending message uncollected by client '%s'.",
                    name
                )
            self._service.outgoing_messages[name] = message

    async def send_message(
            self,
            client: str,
            action: str,
            params: Dict[str, Any]
        ) -> None:
        dump = json.dumps(params, default=json_pack)
        message = {"action": action, "params": dump}
        clients = {d["alias"] for d in self._service.registered_users.values()}
        if client not in clients:
            raise KeyError(f"Unkown destinatory client '{client}'.")
        if client in self._service.outgoing_messages:
            self.logger.warning(
                "Overwriting pending message uncollected by client '%s'.",
                client
            )
        self._service.outgoing_messages[client] = message

    async def wait_for_messages(
            self,
        ) -> Dict[str, Dict[str, Any]]:
        # Wait for all clients to have posted a message.
        while len(self._service.incoming_messages) < self._service.nb_clients:
            await asyncio.sleep(1)  # past: self.heartbeat
        # Grab the (already-deserialized) messages and return them.
        incoming_messages = self._service.incoming_messages
        self._service.incoming_messages = {}
        return incoming_messages
