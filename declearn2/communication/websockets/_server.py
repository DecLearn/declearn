# coding: utf-8

"""Server-side communication endpoint implementation using WebSockets."""

import asyncio
import os
import ssl
from typing import Optional

import websockets as ws
from websockets.server import WebSocketServer, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from declearn2.communication.api import Server
from declearn2.utils import get_logger, register_type


ADD_HEADER = False  # revise: drop this constant (choose a behaviour)


@register_type(name="websockets", group="Server")
class WebsocketsServer(Server):
    """Server-side communication endpoint using WebSockets."""

    logger = get_logger("websockets-server")

    def __init__(
            self,
            host: str = 'localhost',
            port: int = 8765,
            certificate: Optional[str] = None,
            private_key: Optional[str] = None,
            password: Optional[str] = None,
            loop: Optional[asyncio.AbstractEventLoop] = None,
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
        loop: asyncio.AbstractEventLoop or None, default=None
            An asyncio event loop to use.
            If None, use `asyncio.get_event_loop()`.
        """
        # inherited signature; pylint: disable=too-many-arguments
        # Assign attributes and set up optional SSL context.
        super().__init__(host, port, loop=loop)
        self.ssl_context = self._setup_ssl(certificate, private_key, password)
        # Create a server attribute slot.
        self._server = None  # type: Optional[WebSocketServer]

    @property
    def uri(self) -> str:
        protocol = "ws" if self.ssl_context is None else "wss"
        return f"{protocol}://{self.host}:{self.port}"

    @staticmethod
    def _setup_ssl(
            certificate: Optional[str] = None,
            private_key: Optional[str] = None,
            password: Optional[str] = None,
        ) -> Optional[ssl.SSLContext]:
        """Set up and return an (optional) SSLContext object."""
        if (certificate is None) and (private_key is None):
            return None
        if (certificate is None) or (private_key is None):
            raise ValueError(
                "Both 'certificate' and 'private_key' are required "
                "to set up SSL encryption."
            )
        # Optionally read a password from a file.
        if password and os.path.isfile(password):
            with open(password, mode="r", encoding="utf-8") as file:
                password = file.read().strip("\n")
        # Set up the SSLContext, load certificate information and return.
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certificate, private_key, password)
        return ssl_context

    def start(
            self,
        ) -> None:
        """Start the websockets server."""
        # Set up the websockets connections handling process.
        extra_headers = (
            ws.Headers(Connection="keep-alive")  # type: ignore
            if ADD_HEADER else None
        )
        server = ws.serve(  # type: ignore  # pylint: disable=no-member
            self._handle_connection,
            host=self.host,
            port=self.port,
            logger=self.logger,
            ssl=self.ssl_context,
            extra_headers=extra_headers,
        )
        # Run the websockets server.
        self.logger.info("Server is now starting...")
        self._server = self.loop.run_until_complete(server)

    async def _handle_connection(
            self,
            socket: WebSocketServerProtocol,
        ) -> None:
        """WebSockets handler to manage incoming client connections."""
        self.logger.info("New connection from %s", socket.remote_address)
        try:
            async for message in socket:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                reply = await self.handler.handle_message(message, socket)
                await socket.send(reply.to_string())
                if socket not in self.handler.registered_clients:
                    break
        except (ConnectionClosedOK, ConnectionClosedError) as exc:
            name = self.handler.registered_clients.pop(
                socket, socket.remote_address
            )
            self.logger.error(
                "Connection from client '%s' was closed unexpectedly: %s",
                name, exc
            )
        finally:
            await socket.close()

    def stop(
            self,
        ) -> None:
        """Stop the websockets server and purge information about clients."""
        if self._server is not None:
            self._server.close()
            self._server = None
        self.loop.run_until_complete(self.handler.purge())
