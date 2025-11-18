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

"""Unit tests for gRPC network communication tools.

The tests implemented here only test that communications work as expected,
with and without TLS/SSL use, to exchange Ping requests on the local host,
using either low-level gRPC classes wrapping the declearn-defined protobuf
generated code or high-level declearn GrpcClient/GrpcServer classes.

Tests dealing with more complex methods, Client/Server API enforcement and
proper behaviour in the context of Federated Learning are left to separate
test scripts.
"""

import asyncio
import uuid
from typing import AsyncIterator, Dict, Iterator

import pytest
import pytest_asyncio

try:
    import grpc  # type: ignore
except ModuleNotFoundError:
    pytest.skip("GRPC is unavailable", allow_module_level=True)

from declearn.communication.api.backend.actions import Ping
from declearn.communication.grpc import GrpcClient, GrpcServer
from declearn.communication.grpc._server import load_pem_file
from declearn.communication.grpc.protobufs import message_pb2
from declearn.communication.grpc.protobufs.message_pb2_grpc import (
    MessageBoardServicer,
    MessageBoardStub,
    add_MessageBoardServicer_to_server,
)
from declearn.messaging import Message

#################################################################
# 0. Set up pytest fixtures to avoid redundant code in tests

HOST = "localhost"
PORT = 50051
SERVER_URI = f"{HOST}:{PORT}"


class StubMessage(Message):
    """Minimal stub Message subclass."""

    typekey = f"stub-{uuid.uuid4()}"


class FakeMessageBoard(MessageBoardServicer):
    """Minimal MessageBoard implementation to test the connection."""

    def ping(
        self,
        request: message_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> message_pb2.Empty:
        return message_pb2.Empty()

    def send(
        self,
        request: message_pb2.Message,
        context: grpc.ServicerContext,
    ) -> Iterator[message_pb2.Message]:
        yield message_pb2.Message(message=Ping().to_string())


@pytest_asyncio.fixture(name="insecure_grpc_server")
async def insecure_grpc_server_fixture() -> AsyncIterator[grpc.aio.Server]:
    """Create, start and return a grpc Server with unsecured communications."""
    server = grpc.aio.server()
    mboard = FakeMessageBoard()
    add_MessageBoardServicer_to_server(mboard, server)  # type: ignore
    server.add_insecure_port(SERVER_URI)
    await server.start()
    yield server
    await server.stop(0)


@pytest_asyncio.fixture(name="secure_grpc_server")
async def secure_grpc_server_fixture(
    ssl_cert: Dict[str, str],
) -> AsyncIterator[grpc.aio.Server]:
    """Create, start and return a grpc Server with secured communications."""
    server = grpc.aio.server()
    mboard = FakeMessageBoard()
    add_MessageBoardServicer_to_server(mboard, server)  # type: ignore
    pkey = load_pem_file(ssl_cert["server_pkey"])
    cert = load_pem_file(ssl_cert["server_cert"])
    credentials = grpc.ssl_server_credentials([(pkey, cert)])
    server.add_secure_port(SERVER_URI, credentials)
    await server.start()
    yield server
    await server.stop(0)


@pytest_asyncio.fixture(name="insecure_grpc_client")
async def insecure_grpc_client_fixture() -> AsyncIterator[MessageBoardStub]:
    """Create and return MessageBoardStub with unsecured communications."""
    channel = grpc.aio.insecure_channel(SERVER_URI)
    yield MessageBoardStub(channel)  # type: ignore
    await channel.close()


@pytest_asyncio.fixture(name="secure_grpc_client")
async def secure_grpc_client_fixture(
    ssl_cert: Dict[str, str],
) -> AsyncIterator[MessageBoardStub]:
    """Create and return MessageBoardStub with secured communications."""
    certificate = load_pem_file(ssl_cert["client_cert"])
    credentials = grpc.ssl_channel_credentials(certificate)
    channel = grpc.aio.secure_channel(SERVER_URI, credentials)
    yield MessageBoardStub(channel)  # type: ignore
    await channel.close()


@pytest_asyncio.fixture(name="insecure_declearn_server")
async def insecure_declearn_server_fixture() -> AsyncIterator[GrpcServer]:
    """Create and return a GrpcServer with unsecured communications."""
    server = GrpcServer(host=HOST, port=PORT, heartbeat=0.1)
    async with server:
        yield server


@pytest_asyncio.fixture(name="secure_declearn_server")
async def secure_declearn_server_fixture(
    ssl_cert: Dict[str, str],
) -> AsyncIterator[GrpcServer]:
    """Create and return a GrpcServer with secured communications."""
    server = GrpcServer(
        host=HOST,
        port=PORT,
        certificate=ssl_cert["server_cert"],
        private_key=ssl_cert["server_pkey"],
        heartbeat=0.1,
    )
    async with server:
        yield server


@pytest_asyncio.fixture(name="insecure_declearn_client")
async def insecure_declearn_client_fixture() -> AsyncIterator[GrpcClient]:
    """Create and return a GrpcClient with unsecured communications."""
    client = GrpcClient(server_uri=SERVER_URI, name="client")
    await client.start()
    yield client
    await client.stop()


@pytest_asyncio.fixture(name="secure_declearn_client")
async def secure_declearn_client_fixture(
    ssl_cert: Dict[str, str],
) -> AsyncIterator[GrpcClient]:
    """Create and return a GrpcClient with secured communications."""
    client = GrpcClient(
        server_uri=SERVER_URI,
        name="client",
        certificate=ssl_cert["client_cert"],
    )
    await client.start()
    yield client
    await client.stop()


#################################################################
# 1. Test the generated server and client classes


@pytest.mark.asyncio
async def test_message_pb2_grpc_server_insecure(
    insecure_grpc_server: grpc.Server,
    insecure_grpc_client: MessageBoardStub,
) -> None:
    """Unit test for minimal gRPC unsecured communications."""
    # fixture; pylint: disable=unused-argument
    stub = insecure_grpc_client
    response = await stub.ping(message_pb2.Empty())
    assert isinstance(response, message_pb2.Empty)


@pytest.mark.asyncio
async def test_message_pb2_grpc_server_secure_successful_on_secure_channel(
    secure_grpc_server: grpc.Server,
    secure_grpc_client: MessageBoardStub,
) -> None:
    """Unit test for minimal gRPC secured communications."""
    # fixture; pylint: disable=unused-argument
    stub = secure_grpc_client
    response = await stub.ping(message_pb2.Empty())
    assert isinstance(response, message_pb2.Empty)


@pytest.mark.asyncio
async def test_message_pb2_grpc_server_secure_unsuccessful_on_insecure_channel(
    secure_grpc_server: grpc.Server,
    insecure_grpc_client: MessageBoardStub,
) -> None:
    """Unit test for gRPC failure due to unproper security settings."""
    # fixture; pylint: disable=unused-argument
    stub = insecure_grpc_client
    with pytest.raises(grpc.aio.AioRpcError):
        await stub.ping(message_pb2.Empty())


#################################################################
# 2. Test the gRPC server wrapped in declearn Server class


@pytest.mark.asyncio
async def test_grpc_server_insecure(
    insecure_declearn_server: GrpcServer,
    insecure_grpc_client: MessageBoardStub,
) -> None:
    """Unit test for minimal unsecured GrpcServer use."""
    # fixture; pylint: disable=unused-argument
    stub = insecure_grpc_client
    response = await stub.ping(message_pb2.Empty())
    assert isinstance(response, message_pb2.Empty)


@pytest.mark.asyncio
async def test_grpc_server_secure_successful_on_secure_channel(
    secure_declearn_server: GrpcServer,
    secure_grpc_client: MessageBoardStub,
) -> None:
    """Unit test for minimal unsecured GrpcServer use."""
    # fixture; pylint: disable=unused-argument
    stub = secure_grpc_client
    response = await stub.ping(message_pb2.Empty())
    assert isinstance(response, message_pb2.Empty)


@pytest.mark.asyncio
async def test_grpc_server_secure_unsuccessful_on_insecure_channel(
    secure_declearn_server: GrpcServer,
    insecure_grpc_client: MessageBoardStub,
) -> None:
    """Unit test for GrpcServer failure due to unproper security settings."""
    # fixture; pylint: disable=unused-argument
    stub = insecure_grpc_client
    with pytest.raises(grpc.aio.AioRpcError):
        await stub.ping(message_pb2.Empty())


#################################################################
# 3. Test the gRPC channel wrapped in declearn Client class


@pytest.mark.asyncio
async def test_client_with_insecure_grpc_server(
    insecure_grpc_server: grpc.Server,
    insecure_declearn_client: GrpcClient,
) -> None:
    """Unit test for minimal unsecured GrpcClient use."""
    # fixture; pylint: disable=unused-argument
    client = insecure_declearn_client
    await client.send_message(StubMessage())


@pytest.mark.asyncio
async def test_secure_client_with_secure_grpc_server(
    secure_grpc_server: grpc.Server,
    secure_declearn_client: GrpcClient,
) -> None:
    """Unit test for minimal secured GrpcClient use."""
    # fixture; pylint: disable=unused-argument
    client = secure_declearn_client
    await client.send_message(StubMessage())


@pytest.mark.asyncio
async def test_insecure_client_with_secure_grpc_server_fails(
    secure_grpc_server: grpc.Server,
    insecure_declearn_client: GrpcClient,
) -> None:
    """Unit test for GrpcClient failure due to unproper security settings."""
    # fixture; pylint: disable=unused-argument
    client = insecure_declearn_client
    with pytest.raises(grpc.aio.AioRpcError):
        await client.send_message(StubMessage())


#################################################################
# 4. Test the declearn Server and Client classes together


@pytest.mark.asyncio
async def test_client_with_insecure_server(
    insecure_declearn_server: GrpcServer,
    insecure_declearn_client: GrpcClient,
) -> None:
    """Unit test for minimal unsecured GrpcServer/GrpcClient use."""
    # fixture; pylint: disable=unused-argument
    client = insecure_declearn_client
    server = insecure_declearn_server
    await asyncio.gather(
        server.wait_for_clients(1, timeout=5), client.register({})
    )
    await client.send_message(StubMessage())


@pytest.mark.asyncio
async def test_secure_client_with_secure_server(
    secure_declearn_server: GrpcServer,
    secure_declearn_client: GrpcClient,
) -> None:
    """Unit test for minimal secured GrpcServer/GrpcClient use."""
    # fixture; pylint: disable=unused-argument
    client = secure_declearn_client
    server = secure_declearn_server
    await asyncio.gather(
        server.wait_for_clients(1, timeout=5), client.register({})
    )
    await client.send_message(StubMessage())


@pytest.mark.asyncio
async def test_insecure_client_with_secure_server_fails(
    secure_declearn_server: GrpcServer,
    insecure_declearn_client: GrpcClient,
) -> None:
    """Unit test for declearn-gRPC failure due to security asymmetry (1/2)."""
    # fixture; pylint: disable=unused-argument
    client = insecure_declearn_client
    with pytest.raises(grpc.aio.AioRpcError):
        await client.send_message(StubMessage())


@pytest.mark.asyncio
async def test_secure_client_with_insecure_server_fails(
    insecure_declearn_server: GrpcServer,
    secure_declearn_client: GrpcClient,
) -> None:
    """Unit test for declearn-gRPC failure due to security asymmetry (2/2)."""
    # fixture; pylint: disable=unused-argument
    client = secure_declearn_client
    with pytest.raises(grpc.aio.AioRpcError):
        await client.send_message(StubMessage())
