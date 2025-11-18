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

"""Client-side communication endpoint implementation using WebSockets."""

import asyncio
import logging
import ssl
from typing import Any, ClassVar, Dict, Optional, Union

import websockets as ws
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from declearn.communication.api import NetworkClient
from declearn.communication.websockets._tools import (
    receive_websockets_message,
    send_websockets_message,
)

__all__ = [
    "WebsocketsClient",
]


class WebsocketsClient(NetworkClient):
    """Client-side communication endpoint using WebSockets."""

    protocol: ClassVar[str] = "websockets"

    # pylint: disable-next=too-many-positional-arguments
    def __init__(
        self,
        server_uri: str,
        name: str,
        certificate: Optional[str] = None,
        logger: Union[logging.Logger, str, None] = None,
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
        logger: logging.Logger or str or None, default=None,
            Logger to use, or name of a logger to set up using
            `declearn.utils.get_logger`. If None, use `type(self)-name`.
        headers: dict[str, str] or None, default=None
            Optional non-default HTTP headers to use when connecting to
            the server, during the handshake. This may be required when
            connecting through a proxy. For further information, see
            RFC 6455 (https://tools.ietf.org/html/rfc6455#section-1.2).
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(server_uri, name, certificate, logger)
        self.headers = headers
        self._socket: Optional[WebSocketClientProtocol] = None

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

    async def start(self) -> None:
        # false-positives; pylint: disable=no-member
        if not (self._socket is None or self._socket.closed):
            self.logger.info("Client is already connected.")
            return None
        # Set up parameters for `websockets.connect`.
        kwargs = {
            "uri": self.server_uri,
            "logger": self.logger,
            "ssl": self._ssl,
            "extra_headers": (
                ws.Headers(**self.headers)  # type: ignore
                if self.headers
                else None
            ),
            "ping_timeout": None,  # disable timeout on keep-alive pings
        }
        # If connection fails, retry after 1 second - at most 10 times.
        idx = 0
        while True:
            idx += 1
            try:
                self._socket = await ws.connect(**kwargs)  # type: ignore
            except (OSError, asyncio.TimeoutError) as err:
                self.logger.info(
                    "Connection failed (attempt %s/10): %s", idx, err
                )
                if idx == 10:
                    raise err
                await asyncio.sleep(1)
            else:
                self.logger.info("Connected to the server.")
                break

    async def stop(self) -> None:
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    async def _send_message(
        self,
        message: str,
    ) -> str:
        """Send a message to the server and return the obtained reply."""
        if self._socket is None:
            raise RuntimeError("Cannot communicate while not connected.")
        await send_websockets_message(message, self._socket)
        answer = await self._socket.recv()
        return await receive_websockets_message(
            message=answer, socket=self._socket, allow_chunks=True
        )

    async def register(
        self,
        data_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        try:
            return await super().register(data_info)
        except (ConnectionClosedOK, ConnectionClosedError) as err:
            self.logger.error("Connection closed during registration: %s", err)
            self.logger.info("Reconnecting to the server.")
            await self.stop()
            await self.start()
            return False
