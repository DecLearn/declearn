# coding: utf-8

"""Client-side communication endpoint implementation using WebSockets."""

import asyncio
import json
import ssl
from typing import Any, Dict, Optional, Tuple

import websockets as ws
from typing_extensions import Literal  # future: import from typing (Py>=3.8)
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from declearn2.communication.api import Client, flags
from declearn2.utils import get_logger, json_pack, json_unpack, register_type


register_type(name="websockets", group="Client")
class WebsocketsClient(Client):
    """Client-side communication endpoint using WebSockets."""

    logger = get_logger("websockets-client")

    def __init__(
            self,
            server_uri: str,
            name: str,
            certificate: Optional[str] = None,
            loop: Optional[asyncio.AbstractEventLoop] = None,
            headers: Optional[Dict[str, str]] = None,
        ) -> None:
        """Instantiate the client-side WebSockets communications handler.

        Parameters
        ----------
        server_uri: str
            Public uri of the WebSockets server to which this client is
            to connect (e.g. "wss://127.0.0.1:8765").
        name: str
            Name of this client, reported to the server for logging and
            messages' addressing purposes.
        certificate: str or None, default=None,
            Path to a certificate (publickey) PEM file, to use SSL/TLS
            communcations encryption.
        loop: asyncio.AbstractEventLoop or None, default=None
            An asyncio event loop to use.
            If None, use `asyncio.get_event_loop()`.
        headers: dict[str, str] or None, default=None
            Optional non-default HTTP headers to use when connecting to
            the server, during the handshake. This may be required when
            connecting through a proxy. For further information, see
            RFC 6455 (https://tools.ietf.org/html/rfc6455#section-1.2).
        """
        # add one argument; pylint: disable=too-many-arguments
        # Assign attributes. Handle TLS/SSL credentials and HTTP headers.
        super().__init__(server_uri, name, loop=loop)
        self.ssl_context = self._setup_ssl(certificate)
        self.headers = headers
        # Assign a private attribute to handle the connection socket.
        self._socket = None  # type: Optional[WebSocketClientProtocol]

    @staticmethod
    def _setup_ssl(
            certificate: Optional[str] = None,
        ) -> Optional[ssl.SSLContext]:
        """Set up and return an (optional) SSLContext object."""
        if certificate is None:
            return None
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.load_verify_locations(cafile=certificate)
        ssl_context.check_hostname = True  # match the peer cert’s hostname
        ssl_context.post_handshake_auth = True  # for TLS version 3 or higher
        return ssl_context

    def start(
            self
        ) -> None:
        # false-positives; pylint: disable=no-member
        extra_headers = (
            ws.Headers(**self.headers)  # type: ignore
            if self.headers else None
        )
        connect = ws.connect(  # type: ignore
            self.server_uri,
            logger=self.logger,
            ping_interval=None,  # revise: use keep-alive pings?
            ssl=self.ssl_context,
            extra_headers=extra_headers,
        )
        self._socket = self.loop.run_until_complete(connect)

    def stop(
            self
        ) -> None:
        if self._socket is not None:
            self.loop.run_until_complete(self._socket.close())
            self._socket = None

    async def register(
            self,
            data_info: Dict[str, Any],
        ) -> Literal[  # type: ignore
            flags.FLAG_WELCOME, flags.FLAG_REFUSE_CONNECTION
        ]:
        if self._socket is None:
            raise RuntimeError("Cannot register while not connected.")
        # Set up and send the join request.
        message = {
            "type": flags.FIRST_CONNECTION,
            "name": self.name,
            "data_info": data_info,
        }
        request = json.dumps(message, default=json_pack)
        await self._socket.send(request)
        # Wait for the server's reply and return it.
        try:
            reply = await self._socket.recv()
            assert isinstance(reply, str)
        except (ConnectionClosedOK, ConnectionClosedError) as err:
            self.logger.error("Connection closed during registration: %s", err)
            reply = flags.FLAG_REFUSE_CONNECTION
        if reply not in (flags.FLAG_WELCOME, flags.FLAG_REFUSE_CONNECTION):
            raise ValueError(
                f"Invalid server reply to registration request: '{reply}'."
            )
        return reply

    async def send_message(
            self,
            message: Dict[str, Any],
        ) -> None:
        if self._socket is None:
            raise RuntimeError("Cannot send messages while not connected.")
        string = json.dumps(message, default=json_pack)
        await self._socket.send(string)

    async def check_message(
            self,
        ) -> Tuple[str, Dict[str, Any]]:
        if self._socket is None:
            raise RuntimeError("Cannot receive messages while not connected.")
        string = await self._socket.recv()
        message = json.loads(string, object_hook=json_unpack)
        if message.keys() != {"action", "params"}:
            raise KeyError(
                f"Received a message of unproper format: '{message}'."
            )
        return message["action"], message["params"]
