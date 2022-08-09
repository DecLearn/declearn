# coding: utf-8

"""Server-side communication endpoint implementation using WebSockets."""

import asyncio
import json
import logging
import os
import ssl
from typing import Any, Awaitable, Callable, Dict, Optional

import websockets as ws
from websockets.server import WebSocketServer, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from declearn2.communication.api import Server, flags
from declearn2.utils import json_pack, json_unpack


ADD_HEADER = False  # revise: drop this constant (choose a behaviour)

class WebsocketsServer(Server):
    """Server-side communication endpoint using WebSockets."""

    def __init__(
            self,
            nb_clients: int,
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
        nb_clients: int
            Maximum number of clients that should be accepted.
        host : str, dafault='localhost'
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
        self.nb_clients = nb_clients
        self.host = host
        self.port = port
        self.ssl_context = self._setup_ssl(certificate, private_key, password)
        if loop is None:
            loop = asyncio.get_event_loop()
        self.loop = loop  # type: asyncio.AbstractEventLoop
        # Set up private attributes storing clients information.
        self._clients = {}  # type: Dict[WebSocketServerProtocol, str]
        self._data_info = {}  # type: Dict[str, Dict[str, Any]]
        # Set up private attributes to handle asynchronous awaitables.
        self._running = self.loop.create_future()
        self._running.cancel()  # ensure the server is not marked as running
        #
        self.logger = logging.getLogger('websockets')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self._server = None  # type: Optional[WebSocketServer]

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

    def run_until_complete(
            self,
            task: Callable[[], Awaitable],
        ) -> None:
        self.start()
        try:
            self.loop.run_until_complete(task())
        except asyncio.CancelledError as cerr:
            msg = f"Asyncio error while running the server: {cerr}"
            self.logger.info(msg)
        finally:
            self.stop()

    def start(
            self,
        ) -> None:
        """Start the websockets server."""
        # Create a new Future object to await so as to keep connections alive.
        self._running.cancel()
        self._running = self.loop.create_future()
        # Set up the websockets connections handling process.
        extra_headers = (
            ws.Headers(Connection="keep-alive")  # type: ignore
            if ADD_HEADER else None
        )
        server = ws.serve(  # type: ignore  # pylint: disable=no-member
            self._handle_connection,
            host=self.host,
            port=self.port,
            ssl=self.ssl_context,
            extra_headers=extra_headers,
        )
        # Run the websockets server.
        self.logger.info("Server is now starting...")
        self._server = self.loop.run_until_complete(server)

    async def _handle_connection(
            self,
            socket: WebSocketServerProtocol,  # pylint: disable=no-member
        ) -> None:
        """WebSockets handler to manage incoming client connections."""
        # Handle the registration process and decide to accept or reject.
        accept = False
        try:
            accept = await self._handle_registration_request(socket)
        # Log about expected failures in the registration process.
        except (KeyError, ValueError) as err:
            self.logger.error("Error while handling new connection:\n%s", err)
        # Log about connection closing during the registration process.
        except (ConnectionClosedOK, ConnectionClosedError) as err:
            self.logger.error(
                "Connection from client was closed before registration "
                "or rejection could happen:\n%s", err
            )
        # If the client was accepted, maintain the connection indefinitely.
        if accept:
            await self._running
        await socket.close()

    async def _handle_registration_request(
            self,
            socket: WebSocketServerProtocol,  # pylint: disable=no-member
        ) -> bool:
        """Handle the registration request of a new incoming client socket."""
        # Expect an initial message from the new client.
        msg_string = await socket.recv()
        msg = json.loads(msg_string, object_hook=json_unpack)
        # If the message does not conform to expected format, raise.
        for key in ("type", "name", "data_info"):
            if key not in msg:
                raise KeyError(
                    f"Missing required key in join-request message: '{key}'."
                )
        # Otherwise, log the participation request.
        self.logger.info(
            "Received a new connection from client '%s' with ip address '%s'.",
            msg['name'], socket.remote_address
        )
        # If the message has an unproper action flag, warn and exit.
        if not msg["type"] == flags.FIRST_CONNECTION:
            raise ValueError(f"Unproper first-message flag '{msg['type']}'.")
        # If there is room for more participants, register the client.
        if len(self._clients) < self.nb_clients:
            self.logger.info(
                "Registering client '%s' for training.", msg['name']
            )
            await socket.send(flags.FLAG_WELCOME)
            self._clients[socket] = msg['name']
            self._data_info[msg['name']] = msg['data_info']
            return True
        # Otherwise, deny to register it.
        self.logger.info("Rejecting request from client '%s'.", msg['name'])
        await socket.send(flags.FLAG_REFUSE_CONNECTION)
        return False

    def stop(
            self,
        ) -> None:
        """Stop the websockets server and purge information about clients."""
        if not self._running.done():
            self._running.cancel()
        if self._server is not None:
            self._server.close()
            self._server = None
        self._clients = {}
        self._data_info = {}

    async def wait_for_clients(
            self,
        ) -> Dict[str, Dict[str, Any]]:
        self.logger.info("Waiting for clients to register for training...")
        number = 0
        while len(self._clients) < self.nb_clients:
            await asyncio.sleep(1)  # past: self.hearbeat
            dropped = []  # type: List[WebSocketServerProtocol]
            for socket, name in self._clients.items():
                try:
                    await socket.ping()
                except (ConnectionClosedOK, ConnectionClosedError):
                    self.logger.info(
                        "Client '%s' disconnected while waiting for "
                        "participants.", name
                    )
                    dropped.append(socket)
            for socket in dropped:
                self._clients.pop(socket)
            if dropped or (len(self._clients) != number):
                self.logger.info(
                    "Now waiting for %s additional participants.",
                    self.nb_clients - len(self._clients)
                )
            number = len(self._clients)
        return self._data_info.copy()

    def broadcast_message(
            self,
            action: str,
            params: Dict[str, Any],
        ) -> None:
        if self._running.done():
            raise RuntimeError(
                "Cannot broadcast messages while the server is not running."
            )
        dat = {"action": action, "params": params}
        msg = json.dumps(dat, default=json_pack)
        # pylint: disable=no-member
        ws.broadcast(list(self._clients), msg)  # type: ignore

    async def wait_for_messages(
            self,
        ) -> Dict[str, Dict[str, Any]]:
        if self._running.done():
            raise RuntimeError(
                "Cannot await messages while the server is not running."
            )
        # Await for each client to have sent a message.
        routines = [socket.recv() for socket in self._clients]
        received = await asyncio.gather(*routines, return_exceptions=True)
        # Deserialize messages' content, and raise exceptions if any.
        messages = {}  # type: Dict[str, Dict[str, Any]]
        for name, message in zip(self._clients.values(), received):
            if isinstance(message, Exception):
                raise message
            messages[name] = json.loads(message, object_hook=json_unpack)
        return messages
