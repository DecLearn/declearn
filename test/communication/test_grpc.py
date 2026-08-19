# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Iterator, Tuple

import pytest

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

HOST = "localhost"
DYNAMIC_PORT = 0
# Note: Port 0 means gRPC dynamic port allocation, which is encouraged in
# testing contexts to avoid network collisions.
# In the past, the same port 50051 was used in all tests but lead to test
# failures in specific contexts.


#################################################################
# 0. Set up utility classes and context managers to avoid redundant


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
        yield message_pb2.Message(message=Ping().serialize())


@asynccontextmanager
async def insecure_grpc_server() -> AsyncGenerator[
    Tuple[grpc.aio.Server, int], None
]:
    """Create, start and return a grpc Server with unsecured communications,
    and the port used by this server.
    """
    server = grpc.aio.server()
    mboard = FakeMessageBoard()
    add_MessageBoardServicer_to_server(mboard, server)  # type: ignore
    port = server.add_insecure_port(f"{HOST}:{DYNAMIC_PORT}")
    await server.start()
    try:
        yield server, port
    finally:
        await server.stop(0)


@asynccontextmanager
async def secure_grpc_server(
    ssl_cert: Dict[str, str],
) -> AsyncGenerator[Tuple[grpc.aio.Server, int], None]:
    """Create, start and return a grpc Server with secured communications,
    and the port used by this server.
    """
    server = grpc.aio.server()
    mboard = FakeMessageBoard()
    add_MessageBoardServicer_to_server(mboard, server)  # type: ignore
    pkey = load_pem_file(ssl_cert["server_pkey"])
    cert = load_pem_file(ssl_cert["server_cert"])
    credentials = grpc.ssl_server_credentials([(pkey, cert)])
    port = server.add_secure_port(f"{HOST}:{DYNAMIC_PORT}", credentials)
    await server.start()
    try:
        yield server, port
    finally:
        await server.stop(0)


@asynccontextmanager
async def insecure_grpc_client(
    port: int,
) -> AsyncGenerator[MessageBoardStub, None]:
    """Create and return MessageBoardStub with unsecured communications."""
    channel = grpc.aio.insecure_channel(f"{HOST}:{port}")
    try:
        yield MessageBoardStub(channel)  # type: ignore
    finally:
        await channel.close()


@asynccontextmanager
async def secure_grpc_client(
    port: int,
    ssl_cert: Dict[str, str],
) -> AsyncGenerator[MessageBoardStub, None]:
    """Create and return MessageBoardStub with secured communications."""
    certificate = load_pem_file(ssl_cert["client_cert"])
    credentials = grpc.ssl_channel_credentials(certificate)
    channel = grpc.aio.secure_channel(f"{HOST}:{port}", credentials)
    try:
        yield MessageBoardStub(channel)  # type: ignore
    finally:
        await channel.close()


@asynccontextmanager
async def insecure_declearn_server() -> AsyncGenerator[GrpcServer, None]:
    """Create and return a GrpcServer with unsecured communications,
    and the port used by this server.
    """
    server = GrpcServer(host=HOST, port=DYNAMIC_PORT, heartbeat=0.1)
    async with server:
        yield server, server.port


@asynccontextmanager
async def secure_declearn_server(
    ssl_cert: Dict[str, str],
) -> AsyncGenerator[Tuple[GrpcServer, int], None]:
    """Create and return a GrpcServer with secured communications,
    and the port used by this server.
    """
    server = GrpcServer(
        host=HOST,
        port=DYNAMIC_PORT,
        certificate=ssl_cert["server_cert"],
        private_key=ssl_cert["server_pkey"],
        heartbeat=0.1,
    )
    async with server:
        yield server, server.port


@asynccontextmanager
async def insecure_declearn_client(
    port: int,
) -> AsyncGenerator[GrpcClient, None]:
    """Create and return a GrpcClient with unsecured communications."""
    client = GrpcClient(server_uri=f"{HOST}:{port}", name="client")
    await client.start()
    try:
        yield client
    finally:
        await client.stop()


@asynccontextmanager
async def secure_declearn_client(
    port: int,
    ssl_cert: Dict[str, str],
) -> AsyncGenerator[GrpcClient, None]:
    """Create and return a GrpcClient with secured communications."""
    client = GrpcClient(
        server_uri=f"{HOST}:{port}",
        name="client",
        certificate=ssl_cert["client_cert"],
    )
    await client.start()
    try:
        yield client
    finally:
        await client.stop()


#################################################################
# 1. Test the generated server and client classes


@pytest.mark.asyncio
async def test_message_pb2_grpc_server_insecure() -> None:
    """Unit test for minimal gRPC unsecured communications."""
    async with insecure_grpc_server() as (_, port):
        async with insecure_grpc_client(port) as stub:
            response = await stub.ping(message_pb2.Empty())
            assert isinstance(response, message_pb2.Empty)


@pytest.mark.asyncio
async def test_message_pb2_grpc_server_secure_successful_on_secure_channel(
    ssl_cert: Dict[str, str],
) -> None:
    """Unit test for minimal gRPC secured communications."""
    async with secure_grpc_server(ssl_cert) as (_, port):
        async with secure_grpc_client(port, ssl_cert) as stub:
            response = await stub.ping(message_pb2.Empty())
            assert isinstance(response, message_pb2.Empty)


@pytest.mark.asyncio
async def test_message_pb2_grpc_server_secure_unsuccessful_on_insecure_channel(
    ssl_cert: Dict[str, str],
) -> None:
    """Unit test for gRPC failure due to unproper security settings."""
    async with secure_grpc_server(ssl_cert) as (_, port):
        async with insecure_grpc_client(port) as stub:
            with pytest.raises(grpc.aio.AioRpcError):
                await stub.ping(message_pb2.Empty())


#################################################################
# 2. Test the gRPC server wrapped in declearn Server class


@pytest.mark.asyncio
async def test_grpc_server_insecure() -> None:
    """Unit test for minimal unsecured GrpcServer use."""
    async with insecure_declearn_server() as (_, port):
        async with insecure_grpc_client(port) as stub:
            response = await stub.ping(message_pb2.Empty())
            assert isinstance(response, message_pb2.Empty)


@pytest.mark.asyncio
async def test_grpc_server_secure_successful_on_secure_channel(
    ssl_cert: Dict[str, str],
) -> None:
    """Unit test for minimal secured GrpcServer use."""
    async with secure_declearn_server(ssl_cert) as (_, port):
        async with secure_grpc_client(port, ssl_cert) as stub:
            response = await stub.ping(message_pb2.Empty())
            assert isinstance(response, message_pb2.Empty)


@pytest.mark.asyncio
async def test_grpc_server_secure_unsuccessful_on_insecure_channel(
    ssl_cert: Dict[str, str],
) -> None:
    """Unit test for GrpcServer failure due to unproper security settings."""
    async with secure_declearn_server(ssl_cert) as (_, port):
        async with insecure_grpc_client(port) as stub:
            with pytest.raises(grpc.aio.AioRpcError):
                await stub.ping(message_pb2.Empty())


#################################################################
# 3. Test the gRPC channel wrapped in declearn Client class


@pytest.mark.asyncio
async def test_client_with_insecure_grpc_server() -> None:
    """Unit test for minimal unsecured GrpcClient use."""
    async with insecure_grpc_server() as (server, port):
        async with insecure_declearn_client(port) as client:
            await client.send_message(StubMessage())


@pytest.mark.asyncio
async def test_secure_client_with_secure_grpc_server(
    ssl_cert: Dict[str, str],
) -> None:
    """Unit test for minimal secured GrpcClient use."""
    async with secure_grpc_server(ssl_cert) as (_, port):
        async with secure_declearn_client(port, ssl_cert) as client:
            await client.send_message(StubMessage())


@pytest.mark.asyncio
async def test_insecure_client_with_secure_grpc_server_fails(
    ssl_cert: Dict[str, str],
) -> None:
    """Unit test for GrpcClient failure due to unproper security settings."""
    async with secure_grpc_server(ssl_cert) as (_, port):
        async with insecure_declearn_client(port) as client:
            with pytest.raises(grpc.aio.AioRpcError):
                await client.send_message(StubMessage())


#################################################################
# 4. Test the declearn Server and Client classes together


@pytest.mark.asyncio
async def test_client_with_insecure_server() -> None:
    """Unit test for minimal unsecured GrpcServer/GrpcClient use."""
    async with insecure_declearn_server() as (server, port):
        async with insecure_declearn_client(port) as client:
            await asyncio.gather(
                server.wait_for_clients(1, timeout=5), client.register()
            )
            await client.send_message(StubMessage())


@pytest.mark.asyncio
async def test_secure_client_with_secure_server(
    ssl_cert: Dict[str, str],
) -> None:
    """Unit test for minimal secured GrpcServer/GrpcClient use."""
    async with secure_declearn_server(ssl_cert) as (server, port):
        async with secure_declearn_client(port, ssl_cert) as client:
            await asyncio.gather(
                server.wait_for_clients(1, timeout=5), client.register()
            )
            await client.send_message(StubMessage())


@pytest.mark.asyncio
async def test_insecure_client_with_secure_server_fails(
    ssl_cert: Dict[str, str],
) -> None:
    """Unit test for declearn-gRPC failure due to security asymmetry (1/2)."""
    async with secure_declearn_server(ssl_cert) as (_, port):
        async with insecure_declearn_client(port) as client:
            with pytest.raises(grpc.aio.AioRpcError):
                await client.send_message(StubMessage())


@pytest.mark.asyncio
async def test_secure_client_with_insecure_server_fails(
    ssl_cert: Dict[str, str],
) -> None:
    """Unit test for declearn-gRPC failure due to security asymmetry (2/2)."""
    async with insecure_declearn_server() as (_, port):
        async with secure_declearn_client(port, ssl_cert) as client:
            with pytest.raises(grpc.aio.AioRpcError):
                await client.send_message(StubMessage())
