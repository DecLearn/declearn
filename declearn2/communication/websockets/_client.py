# coding: utf-8

"""Client-side communication endpoint implementation using WebSockets."""

import ssl
from typing import Any, Dict, Optional

import websockets as ws
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from declearn2.communication.api import Client
from declearn2.communication.messaging import (
    Message, parse_message_from_string
)
from declearn2.utils import get_logger, register_type


@register_type(name="websockets", group="Client")
class WebsocketsClient(Client):
    """Client-side communication endpoint using WebSockets."""

    logger = get_logger("websockets-client")

    def __init__(
            self,
            server_uri: str,
            name: str,
            certificate: Optional[str] = None,
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
        headers: dict[str, str] or None, default=None
            Optional non-default HTTP headers to use when connecting to
            the server, during the handshake. This may be required when
            connecting through a proxy. For further information, see
            RFC 6455 (https://tools.ietf.org/html/rfc6455#section-1.2).
        """
        super().__init__(server_uri, name, certificate)
        self.headers = headers
        self._socket = None  # type: Optional[WebSocketClientProtocol]

    @staticmethod
    def _setup_ssl_context(
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

    async def start(
            self
        ) -> None:
        # false-positives; pylint: disable=no-member
        if not (self._socket is None or self._socket.closed):
            self.logger.info("Client is already connected.")
            return None
        extra_headers = (
            ws.Headers(**self.headers)  # type: ignore
            if self.headers else None
        )
        self._socket = await ws.connect(  # type: ignore
            self.server_uri,
            logger=self.logger,
            ping_interval=None,  # revise: use keep-alive pings?
            ssl=self._ssl,
            extra_headers=extra_headers,
        )

    async def stop(
            self
        ) -> None:
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    async def _send_message(
            self,
            message: Message,
        ) -> Message:
        """Send a message to the server and return the obtained reply."""
        if self._socket is None:
            raise RuntimeError("Cannot communicate while not connected.")
        string = message.to_string()
        await self._socket.send(string)
        reply = await self._socket.recv()
        if isinstance(reply, bytes):
            reply = reply.decode("utf-8")
        return parse_message_from_string(reply)

    async def register(
            self,
            data_info: Dict[str, Any],
        ) -> bool:
        try:
            return await super().register(data_info)
        except (ConnectionClosedOK, ConnectionClosedError) as err:
            self.logger.error("Connection closed during registration: %s", err)
            return False
