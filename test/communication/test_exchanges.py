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

"""Minimal unit tests for declearn.communication network endpoint classes.

The tests implemented here gradually assess that a NetworkServer and one
or multiple properly-configured NetworkClient instances can connect with
each other and exchange information over the localhost using the actual
protocol they rely upon (as opposed to assessing classes' behaviors with
mock communication I/O).

Here, tests are run incrementally that assess:
    - the possibility for connections to be set up
    - the possibility for a server to perform clients' registration
    - the possibility to exchange messages in both directions between
      a server and registered clients

All available communication protocols are used, with or without SSL.
In the last tier of tests, unencrypted communications stop being tested
(notably because they can fail on the local host due to clients _not_
being identified as distinct by some protocols, such as gRPC) and tests
are run with either a single or three clients at once.
"""

import asyncio
import secrets
from typing import AsyncIterator, Dict, List, Optional, Tuple

import pytest
import pytest_asyncio

from declearn import messaging
from declearn.communication import (
    build_client,
    build_server,
    list_available_protocols,
)
from declearn.communication.api import NetworkClient, NetworkServer

### 1. Test that connections can be properly set up.


@pytest_asyncio.fixture(name="server")
async def server_fixture(
    protocol: str,
    ssl_cert: Dict[str, str],
    ssl: bool,
) -> AsyncIterator[NetworkServer]:
    """Fixture to provide with an instantiated and started NetworkServer."""
    server = build_server(
        protocol=protocol,
        host="127.0.0.1",  # truly "localhost", but fails on the CI otherwise
        port=8765,
        certificate=ssl_cert["server_cert"] if ssl else None,
        private_key=ssl_cert["server_pkey"] if ssl else None,
        heartbeat=0.1,  # fasten tests by setting a low heartbeat
    )
    async with server:
        yield server


def client_from_server(
    server: NetworkServer,
    c_name: str = "client",
    ca_ssl: Optional[str] = None,
) -> NetworkClient:
    """Instantiate a NetworkClient based on a NetworkServer."""
    return build_client(
        protocol=server.protocol,
        server_uri=server.uri.replace("127.0.0.1", "localhost"),
        name=c_name,
        certificate=ca_ssl,
    )


@pytest.mark.parametrize("ssl", [True, False], ids=["ssl", "no_ssl"])
@pytest.mark.parametrize("protocol", list_available_protocols())
@pytest.mark.asyncio
async def test_network_connect(
    server: NetworkServer,
    ssl_cert: Dict[str, str],
    ssl: bool,
) -> None:
    """Test that connections can be set up for a given framework."""
    ca_ssl = ssl_cert["client_cert"] if ssl else None
    client = client_from_server(server, c_name="client", ca_ssl=ca_ssl)
    await client.start()
    await client.stop()


### 2. Test that a client can be registered over network by a server.


@pytest_asyncio.fixture(name="client")
async def client_fixture(
    server: NetworkServer,
    ssl_cert: Dict[str, str],
    ssl: bool,
) -> AsyncIterator[NetworkClient]:
    """Fixture to provide with an instantiated and started NetworkClient."""
    client = client_from_server(
        server,
        c_name="client",
        ca_ssl=ssl_cert["client_cert"] if ssl else None,
    )
    async with client:
        yield client


@pytest.mark.parametrize("ssl", [True, False], ids=["ssl", "no_ssl"])
@pytest.mark.parametrize("protocol", list_available_protocols())
class TestNetworkRegister:
    """Unit tests for client-registration operations."""

    @pytest.mark.asyncio
    async def test_early_request(
        self, server: NetworkServer, client: NetworkClient
    ) -> None:
        """Test that early registration requests are rejected."""
        accepted = await client.register()
        assert not accepted
        assert not server.client_names

    @pytest.mark.asyncio
    async def test_register(
        self, server: NetworkServer, client: NetworkClient
    ) -> None:
        """Test that client registration works properly."""
        output, accepted = await asyncio.gather(
            server.wait_for_clients(1),
            client.register(),
        )
        assert output is None
        assert accepted
        assert server.client_names == {"client"}

    @pytest.mark.asyncio
    async def test_register_late(
        self, server: NetworkServer, client: NetworkClient
    ) -> None:
        """Test that late client registration fails properly."""
        # Wait for clients, with a timeout.
        with pytest.raises(RuntimeError):
            await server.wait_for_clients(timeout=0.1)
        # Try registering after that timeout.
        accepted = await client.register()
        assert not accepted


### 3. Test that a server and its registered clients can exchange messages.


@pytest_asyncio.fixture(name="agents")
async def agents_fixture(
    server: NetworkServer,
    n_clients: int,
    ssl_cert: Dict[str, str],
    ssl: bool,
) -> AsyncIterator[Tuple[NetworkServer, List[NetworkClient]]]:
    """Fixture to provide with a server and pre-registered client(s)."""
    # Instantiate the clients.
    ca_ssl = ssl_cert["client_cert"] if ssl else None
    clients = [
        client_from_server(server, c_name=f"client-{idx}", ca_ssl=ca_ssl)
        for idx in range(n_clients)
    ]
    # Start the clients and have the server register them.
    await asyncio.gather(*[client.start() for client in clients])
    await asyncio.gather(
        server.wait_for_clients(n_clients, timeout=1),
        *[client.register() for client in clients],
    )
    # Yield the server and clients. On exit, stop the clients.
    yield server, clients
    await asyncio.gather(*[client.stop() for client in clients])


@pytest.mark.parametrize("n_clients", [1, 3], ids=["1_client", "3_clients"])
@pytest.mark.parametrize("ssl", [True], ids=["ssl"])
@pytest.mark.parametrize("protocol", list_available_protocols())
class TestNetworkExchanges:
    """Unit tests for messaging-over-network operations.

    Note: the unit tests implemented here are grouped into a single
          larger call, to avoid setup costs, while preserving code
          and failure-information readability (hopefully).
    """

    @pytest.mark.asyncio
    async def test_exchanges(
        self,
        agents: Tuple[NetworkServer, List[NetworkClient]],
    ) -> None:
        """Run all tests with the same fixture-provided agents."""
        await self.clients_to_server(agents)
        await self.server_to_clients_broadcast(agents)
        await self.server_to_clients_individual(agents)
        await self.clients_to_server_large(agents)

    async def clients_to_server(
        self,
        agents: Tuple[NetworkServer, List[NetworkClient]],
    ) -> None:
        """Test that clients can send messages to the server."""
        server, clients = agents
        coros = []
        for idx, client in enumerate(clients):
            msg = messaging.GenericMessage(action="test", params={"idx": idx})
            coros.append(client.send_message(msg))
        protos, *_ = await asyncio.gather(server.wait_for_messages(), *coros)
        assert all(
            isinstance(proto, messaging.SerializedMessage)
            for proto in protos.values()
        )
        messages = {key: proto.deserialize() for key, proto in protos.items()}
        assert messages == {
            c.name: messaging.GenericMessage(action="test", params={"idx": i})
            for i, c in enumerate(clients)
        }

    async def server_to_clients_broadcast(
        self,
        agents: Tuple[NetworkServer, List[NetworkClient]],
    ) -> None:
        """Test that the server can send a shared message to all clients."""
        server, clients = agents
        msg = messaging.GenericMessage(action="test", params={"value": 42})
        send = server.broadcast_message(msg)
        recv = [client.recv_message(timeout=1) for client in clients]
        _, *replies = await asyncio.gather(send, *recv)
        assert all(
            isinstance(reply, messaging.SerializedMessage) for reply in replies
        )
        assert all(reply.deserialize() == msg for reply in replies)

    async def server_to_clients_individual(
        self,
        agents: Tuple[NetworkServer, List[NetworkClient]],
    ) -> None:
        """Test that the server can send individual messages to clients."""
        server, clients = agents
        messages: Dict[str, messaging.Message] = {
            name: messaging.GenericMessage(action="test", params={"idx": idx})
            for idx, name in enumerate(server.client_names)
        }
        send = server.send_messages(messages)
        recv = [client.recv_message(timeout=1) for client in clients]
        _, *replies = await asyncio.gather(send, *recv)
        assert all(
            isinstance(reply, messaging.SerializedMessage) for reply in replies
        )
        assert all(
            reply.deserialize() == messages[client.name]
            for client, reply in zip(clients, replies, strict=False)
        )

    async def clients_to_server_large(
        self,
        agents: Tuple[NetworkServer, List[NetworkClient]],
    ) -> None:
        """Test that the clients can send large messages to the server."""
        server, clients = agents
        coros = []
        large = secrets.token_bytes(2**22).hex()
        for idx, client in enumerate(clients):
            msg = messaging.GenericMessage(
                action="test", params={"idx": idx, "content": large}
            )
            coros.append(client.send_message(msg))
        protos, *_ = await asyncio.gather(server.wait_for_messages(), *coros)
        assert all(
            isinstance(proto, messaging.SerializedMessage)
            for proto in protos.values()
        )
        messages = {key: proto.deserialize() for key, proto in protos.items()}
        assert messages == {
            c.name: messaging.GenericMessage(
                action="test", params={"idx": i, "content": large}
            )
            for i, c in enumerate(clients)
        }
