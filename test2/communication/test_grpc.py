# coding: utf-8

"""Unit tests for gRPC network communication tools.

The tests implemented here only test that communications work as expected,
with and without TLS/SSL use, to exchange Ping requests on the local host,
using either low-level gRPC classes wrapping the declearn-defined protobuf
generated code or high-level declearn GrpcClient/GrpcServer classes.

Tests dealing with more complex methods, Client/Server API enforcement and
proper behaviour in the context of Federated Learning are left to separate
test scripts.
"""

import os
import shlex
import subprocess
import tempfile
from typing import AsyncIterator, Dict, Iterator, Optional, Tuple

import grpc  # type: ignore
import pytest
import pytest_asyncio

from declearn2.communication.grpc._server import load_pem_file
from declearn2.communication.grpc import GrpcClient, GrpcServer
from declearn2.communication.grpc.protobufs.message_pb2 import Empty
from declearn2.communication.grpc.protobufs.message_pb2_grpc import (
    MessageBoardServicer, MessageBoardStub, add_MessageBoardServicer_to_server
)


#################################################################
# -1. TLS/SSL certificates generation function - to be shared?

def generate_ssl_certificates(
        folder: str = ".",
        c_name: str = "localhost",
        password: Optional[str] = None
    ) -> Tuple[str, str, str]:
    """Generate self-signed certificates for localhost testing purposes.

    Parameters
    ----------
    folder: str
        Path to the (ideally, temporary) folder where to create files.
    c_name: str
        CommonName value for the server certificate.
    password: str or None, default=None
        Optional password used to encrypt generated private keys.

    Returns
    -------
    ca_cert: str
        Path to the client-required CA certificate PEM file.
    sv_cert: str
        Path to the server's certificate PEM file.
    sv_priv: str
        Path to the server's private key PEM file (password: "supertoto").
    """
    # Generate self-signed CA certificate and private key.
    ca_priv = os.path.join(folder, "ca-key.pem")
    ca_cert = os.path.join(folder, "ca-cert.pem")
    cmd = (
        "openssl req -x509 -newkey rsa:4096 "
        + f"-keyout {ca_priv} -out {ca_cert} "
        + (f"-passout pass:{password} " if password else "-nodes ")
        + "-sha256 -days 365 "
        + '-subj "/C=FR/L=Lille/O=Inria/OU=Magnet/CN=inria.fr"'
    )
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    # Generate server private key and CSR (certificate signing request).
    sv_priv = os.path.join(folder, "server-key.pem")
    sv_csrq = os.path.join(folder, "server-req.pem")
    cmd = (
        "openssl req -newkey rsa:4096 "
        + f"-keyout {sv_priv} -out {sv_csrq} "
        + (f"-passout pass:{password} " if password else "-nodes ")
        + f'-subj "/C=FR/L=Lille/O=Inria/OU=Magnet/CN={c_name}"'
    )
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    # Generate self-signed server certificate.
    sv_cert = os.path.join(folder, "server-cert.pem")
    cmd = (
        f"openssl x509 -req -in {sv_csrq} -CA {ca_cert} "
        + f"-CAkey {ca_priv} -CAcreateserial -out {sv_cert} "
        + (f"-passin pass:{password} " if password else " ")
        + "-days 30"
    )
    subprocess.run(shlex.split(cmd), check=True, capture_output=True)
    # Return paths that are used in tests.
    return ca_cert, sv_cert, sv_priv


#################################################################
# 0. Set up pytest fixtures to avoid redundant code in tests

HOST = 'localhost'
PORT = 50051
SERVER_URI = f'{HOST}:{PORT}'


class FakeMessageBoard(MessageBoardServicer):
    """Minimal MessageBoard implementation to test the connection."""

    def ping(self, request: Empty, context: grpc.ServicerContext) -> Empty:
        return Empty()


@pytest.fixture(name="ssl_cert", scope="module")
def ssl_cert_fixture() -> Iterator[Dict[str, str]]:
    """Context manager to use temporary self-signed TLS/SSL certificates.

    Return a dict containing both paths to PEM files and loaded
    bytes content of the latter for a client (CA) certificate,
    a server certificate and a server private key.
    """
    with tempfile.TemporaryDirectory() as folder:
        ca_cert, sv_cert, sv_priv = generate_ssl_certificates(folder)
        yield {
            "client_cert": ca_cert,
            "server_cert": sv_cert,
            "server_pkey": sv_priv,
        }


@pytest_asyncio.fixture(name="insecure_grpc_server")
async def insecure_grpc_server_fixture(
    ) -> AsyncIterator[grpc.Server]:
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
    ) -> AsyncIterator[grpc.Server]:
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
async def insecure_grpc_client_fixture(
    ) -> AsyncIterator[MessageBoardStub]:
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


@pytest.fixture(name="insecure_declearn_server")
def insecure_declearn_server_fixture(
    ) -> Iterator[GrpcServer]:
    """Create and return a GrpcServer with unsecured communications."""
    server = GrpcServer(nb_clients=1, host=HOST, port=PORT)
    server.start()
    yield server
    server.stop()


@pytest.fixture(name="secure_declearn_server")
def secure_declearn_server_fixture(
        ssl_cert: Dict[str, str],
    ) -> Iterator[GrpcServer]:
    """Create and return a GrpcServer with secured communications."""
    server = GrpcServer(
        nb_clients=1, host=HOST, port=PORT,
        certificate=ssl_cert["server_cert"],
        private_key=ssl_cert["server_pkey"],
    )
    server.start()
    yield server
    server.stop()


@pytest.fixture(name="insecure_declearn_client")
def insecure_declearn_client_fixture(
    ) -> Iterator[GrpcClient]:
    """Create and return a GrpcClient with unsecured communications."""
    client = GrpcClient(server_uri=SERVER_URI, name="client")
    client.start()
    yield client
    client.stop()


@pytest.fixture(name="secure_declearn_client")
def secure_declearn_client_fixture(
        ssl_cert: Dict[str, str],
    ) -> Iterator[GrpcClient]:
    """Create and return a GrpcClient with secured communications."""
    client = GrpcClient(
        server_uri=SERVER_URI, name="client",
        certificate=ssl_cert["client_cert"],
    )
    client.start()
    yield client
    client.stop()


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
    response = await stub.ping(Empty())
    assert isinstance(response, Empty)


@pytest.mark.asyncio
async def test_message_pb2_grpc_server_secure_successful_on_secure_channel(
        secure_grpc_server: grpc.Server,
        secure_grpc_client: MessageBoardStub,
    ) -> None:
    """Unit test for minimal gRPC secured communications."""
    # fixture; pylint: disable=unused-argument
    stub = secure_grpc_client
    response = await stub.ping(Empty())
    assert isinstance(response, Empty)


@pytest.mark.asyncio
async def test_message_pb2_grpc_server_secure_unsuccessful_on_insecure_channel(
        secure_grpc_server: grpc.Server,
        insecure_grpc_client: MessageBoardStub,
    ) -> None:
    """Unit test for gRPC failure due to unproper security settings."""
    # fixture; pylint: disable=unused-argument
    stub = insecure_grpc_client
    with pytest.raises(grpc.aio.AioRpcError):
        await stub.ping(Empty())


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
    response = await stub.ping(Empty())
    assert isinstance(response, Empty)


@pytest.mark.asyncio
async def test_grpc_server_secure_successful_on_secure_channel(
        secure_declearn_server: GrpcServer,
        secure_grpc_client: MessageBoardStub,
    ) -> None:
    """Unit test for minimal unsecured GrpcServer use."""
    # fixture; pylint: disable=unused-argument
    stub = secure_grpc_client
    response = await stub.ping(Empty())
    assert isinstance(response, Empty)


@pytest.mark.asyncio
async def test_grpc_server_secure_unsuccessful_on_insecure_channel(
        secure_declearn_server: GrpcServer,
        insecure_grpc_client: MessageBoardStub,
    ) -> None:
    """Unit test for GrpcServer failure due to unproper security settings."""
    # fixture; pylint: disable=unused-argument
    stub = insecure_grpc_client
    with pytest.raises(grpc.aio.AioRpcError):
        await stub.ping(Empty())

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
    ping_ok = await client.ping()
    assert ping_ok is True


@pytest.mark.asyncio
async def test_secure_client_with_secure_grpc_server(
        secure_grpc_server: grpc.Server,
        secure_declearn_client: GrpcClient,
    ) -> None:
    """Unit test for minimal secured GrpcClient use."""
    # fixture; pylint: disable=unused-argument
    client = secure_declearn_client
    ping_ok = await client.ping()
    assert ping_ok is True


@pytest.mark.asyncio
async def test_insecure_client_with_secure_grpc_server_fails(
        secure_grpc_server: grpc.Server,
        insecure_declearn_client: GrpcClient,
    ) -> None:
    """Unit test for GrpcClient failure due to unproper security settings."""
    # fixture; pylint: disable=unused-argument
    client = insecure_declearn_client
    with pytest.raises(grpc.aio.AioRpcError):
        await client.ping()


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
    ping_ok = await client.ping()
    assert ping_ok is True


@pytest.mark.asyncio
async def test_secure_client_with_secure_server(
        secure_declearn_server: GrpcServer,
        secure_declearn_client: GrpcClient,
    ) -> None:
    """Unit test for minimal secured GrpcServer/GrpcClient use."""
    # fixture; pylint: disable=unused-argument
    client = secure_declearn_client
    ping_ok = await client.ping()
    assert ping_ok is True


@pytest.mark.asyncio
async def test_insecure_client_with_secure_server_fails(
        secure_declearn_server: GrpcServer,
        insecure_declearn_client: GrpcClient,
    ) -> None:
    """Unit test for declearn-gRPC failure due to security asymmetry (1/2)."""
    # fixture; pylint: disable=unused-argument
    client = insecure_declearn_client
    with pytest.raises(grpc.aio.AioRpcError):
        await client.ping()


@pytest.mark.asyncio
async def test_secure_client_with_insecure_server_fails(
        insecure_declearn_server: GrpcServer,
        secure_declearn_client: GrpcClient,
    ) -> None:
    """Unit test for declearn-gRPC failure due to security asymmetry (2/2)."""
    # fixture; pylint: disable=unused-argument
    client = secure_declearn_client
    with pytest.raises(grpc.aio.AioRpcError):
        await client.ping()
