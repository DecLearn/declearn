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

"""Fake network communication endpoints relying on shared memory objects."""

import asyncio
import contextlib
import logging
import uuid
from typing import (  # fmt: off
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from declearn.communication.api import NetworkClient, NetworkServer
from declearn.communication.api.backend import MessagesHandler
from declearn.messaging import Message, SerializedMessage

__all__ = [
    "MockNetworkClient",
    "MockNetworkServer",
    "setup_mock_network_endpoints",
]


MessageT = TypeVar("MessageT", bound=Message)


HANDLERS: Dict[str, MessagesHandler] = {}


class MockNetworkServer(NetworkServer, register=False):
    """Fake server network communication endpoint using global dictionaries."""

    protocol = "mock"

    # pylint: disable-next=too-many-positional-arguments
    def __init__(  # noqa: PLR0913
        self,
        host: str = "localhost",
        port: int = 8765,
        certificate: Optional[str] = None,
        private_key: Optional[str] = None,
        password: Optional[str] = None,
        heartbeat: float = 0.1,
        logger: Union[logging.Logger, str, None] = None,
    ) -> None:
        # inherited signature; pylint: disable=too-many-arguments
        # abstract parent method; pylint: disable=useless-parent-delegation
        super().__init__(
            host, port, certificate, private_key, password, heartbeat, logger
        )

    @property
    def uri(self) -> str:
        return f"mock://{self.host}:{self.port}"

    @staticmethod
    def _setup_ssl_context(
        certificate: str,
        private_key: str,
        password: Optional[str] = None,
    ) -> bool:
        return bool(certificate)

    async def start(
        self,
    ) -> None:
        if self.uri in HANDLERS:
            raise RuntimeError(f"Address '{self.uri}' already in use.")
        HANDLERS[self.uri] = self.handler

    async def stop(
        self,
    ) -> None:
        HANDLERS.pop(self.uri, None)

    # Force the use of a timeout, to prevent tests from being stuck.

    async def broadcast_message(
        self,
        message: Message,
        clients: Optional[Set[str]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        timeout = 5 if timeout is None else timeout
        await super().broadcast_message(message, clients, timeout)

    async def send_messages(
        self,
        messages: Mapping[str, MessageT],
        timeout: Optional[float] = None,
    ) -> None:
        timeout = 5 if timeout is None else timeout
        await super().send_messages(messages, timeout)

    async def send_message(
        self,
        message: Message,
        client: str,
        timeout: Optional[float] = None,
    ) -> None:
        timeout = 5 if timeout is None else timeout
        await super().send_message(message, client, timeout)

    async def wait_for_messages(
        self,
        clients: Optional[Set[str]] = None,
    ) -> Dict[str, SerializedMessage]:
        coro = super().wait_for_messages(clients)
        return await asyncio.wait_for(coro, timeout=5)


class MockNetworkClient(NetworkClient, register=False):
    """Fake client network communication endpoint using global dictionaries."""

    protocol = "mock"

    def __init__(
        self,
        server_uri: str = "mock://localhost:8765",
        name: str = "client",
        certificate: Optional[str] = None,
        logger: Union[logging.Logger, str, None] = None,
    ) -> None:
        super().__init__(server_uri, name, certificate, logger)
        self._started = False
        self._uuid = str(uuid.uuid4())

    @staticmethod
    def _setup_ssl_context(
        certificate: Optional[str] = None,
    ) -> bool:
        return bool(certificate)

    @property
    def handler(self) -> MessagesHandler:
        """Server's MessageHandler instance, accessed directly."""
        if not self._started:
            raise RuntimeError("Cannot access handler of stopped client.")
        if (handler := HANDLERS.get(self.server_uri, None)) is None:
            raise KeyError(f"Address '{self.server_uri}' no longer found.")
        return handler

    async def start(
        self,
    ) -> None:
        if self.server_uri not in HANDLERS:
            raise KeyError(f"Address '{self.server_uri}' not found.")
        self._started = True

    async def stop(
        self,
    ) -> None:
        self._started = False

    async def _send_message(
        self,
        message: str,
    ) -> str:
        coro = self.handler.handle_message(message, context=self._uuid)
        # Force the use of a timeout, to prevent tests from being stuck.
        action = await asyncio.wait_for(coro, timeout=5)
        return action.to_string()

    async def recv_message(
        self,
        timeout: Optional[float] = None,
    ) -> SerializedMessage:
        # Force the use of a timeout, to prevent tests from being stuck.
        return await super().recv_message(timeout=timeout or 5)


@contextlib.asynccontextmanager
async def setup_mock_network_endpoints(
    n_peers: int,
    port: int = 8765,
) -> AsyncIterator[Tuple[MockNetworkServer, List[MockNetworkClient]]]:
    """Instantiate, start and register mock network communication endpoints.

    This is an async context manager, that returns network endpoints,
    and ensures they are all properly closed upon leaving the context.

    Parameters
    ----------
    n_peers:
        Number of client endpoints to instantiate.
    port:
        Mock port number to use.

    Returns
    -------
    server:
        `MockNetworkServer` instance to which clients are registered.
    clients:
        List of `MockNetworkClient` instances, registered to the server.
    """
    # Instantiate the endpoints.
    server = MockNetworkServer(port=port)
    clients = [
        MockNetworkClient(f"mock://localhost:{port}", name=f"client_{i}")
        for i in range(n_peers)
    ]
    async with contextlib.AsyncExitStack() as stack:
        # Start the endpoints and ensure they will be properly closed.
        await stack.enter_async_context(server)  # type: ignore
        for client in clients:
            await stack.enter_async_context(client)  # type: ignore
        # Register the clients with the server.
        await asyncio.gather(
            server.wait_for_clients(n_peers),
            *[client.register() for client in clients],
        )
        # Yield the started, registered endpoints.
        yield server, clients
