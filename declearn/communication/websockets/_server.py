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

"""Server-side communication endpoint implementation using WebSockets."""

import logging
import os
import ssl
from typing import Optional, Union

import websockets as ws
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.server import WebSocketServer, WebSocketServerProtocol

from declearn.communication.api import NetworkServer
from declearn.communication.websockets._tools import (
    StreamRefusedError,
    receive_websockets_message,
    send_websockets_message,
)

__all__ = [
    "WebsocketsServer",
]


class WebsocketsServer(NetworkServer):
    """Server-side communication endpoint using WebSockets."""

    protocol = "websockets"

    # pylint: disable-next=too-many-positional-arguments
    def __init__(  # noqa: PLR0913
        self,
        host: str = "localhost",
        port: int = 8765,
        certificate: Optional[str] = None,
        private_key: Optional[str] = None,
        password: Optional[str] = None,
        heartbeat: float = 1.0,
        logger: Union[logging.Logger, str, None] = None,
    ) -> None:
        """Instantiate the server-side WebSockets communications handler.

        Parameters
        ----------
        host : str, default='localhost'
            Host name (e.g. IP address) of the server.
        port: int, default=8765
            Communications port to use.
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
        logger: logging.Logger or str or None, default=None,
            Logger to use, or name of a logger to set up with
            `declearn.utils.get_logger`. If None, use `type(self)`.
        """
        # inherited signature; pylint: disable=too-many-arguments
        super().__init__(
            host, port, certificate, private_key, password, heartbeat, logger
        )
        self._server: Optional[WebSocketServer] = None

    @property
    def uri(self) -> str:
        protocol = "ws" if self._ssl is None else "wss"
        return f"{protocol}://{self.host}:{self.port}"

    @staticmethod
    def _setup_ssl_context(
        certificate: str,
        private_key: str,
        password: Optional[str] = None,
    ) -> Optional[ssl.SSLContext]:
        """Set up and return a SSLContext object."""
        # Optionally read a password from a file.
        if password and os.path.isfile(password):
            with open(password, mode="r", encoding="utf-8") as file:
                password = file.read().strip("\n")
        # Set up the SSLContext, load certificate information and return.
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certificate, private_key, password)
        return ssl_context

    async def start(
        self,
    ) -> None:
        """Start the websockets server."""
        # Set up the websockets connections handling process.
        server = ws.serve(  # type: ignore  # pylint: disable=no-member
            self._handle_connection,
            host=self.host,
            port=self.port,
            logger=self.logger,
            ssl=self._ssl,
            ping_timeout=None,  # disable timeout on keep-alive pings
        )
        # Run the websockets server.
        self.logger.info("Server is now starting...")
        self._server = await server

    async def _handle_connection(
        self,
        socket: WebSocketServerProtocol,
    ) -> None:
        """WebSockets handler to manage incoming client connections."""
        self.logger.info("New connection from %s", socket.remote_address)
        try:
            async for frame in socket:
                # Receive the message (covering chunked-message case).
                known = socket in self.handler.registered_clients
                try:
                    message = await receive_websockets_message(
                        frame, socket, allow_chunks=known
                    )
                except StreamRefusedError:
                    self.logger.warning(
                        "Refused a chunks-streaming request from client %s",
                        socket.remote_address,
                    )
                    break
                # Handle the received message and produce an answer.
                reply = await self.handler.handle_message(message, socket)
                await send_websockets_message(reply.to_string(), socket)
                if socket not in self.handler.registered_clients:
                    break
        except (ConnectionClosedOK, ConnectionClosedError) as exc:
            name = self.handler.registered_clients.pop(
                socket, socket.remote_address
            )
            self.logger.error(
                "Connection from client '%s' was closed unexpectedly: %s",
                name,
                exc,
            )
        finally:
            await socket.close()

    async def stop(
        self,
    ) -> None:
        """Stop the websockets server and purge information about clients."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await self.handler.purge()
