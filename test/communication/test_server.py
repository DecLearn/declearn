# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""Unit tests for `declearn.communication.api.NetworkServer` classes."""

import asyncio
from typing import AsyncIterator, Dict
from unittest import mock

import pytest
import pytest_asyncio

from declearn.communication import (
    build_server,
    list_available_protocols,
    messaging,
)
from declearn.communication.api import NetworkServer
from declearn.utils import access_types_mapping, get_logger


SERVER_CLASSES = access_types_mapping("NetworkServer")


@pytest.mark.parametrize("protocol", list_available_protocols())
class TestNetworkServerInit:
    """Unit tests for `declearn.communication.api.NetworkServer` classes.

    This class groups tests that revolve around instantiating
    a server and accessing its attributes and properties.
    """

    def test_registered(self, protocol: str) -> None:
        """Assert that the tested class is properly type-registered."""
        assert protocol in SERVER_CLASSES
        assert issubclass(SERVER_CLASSES[protocol], NetworkServer)
        assert SERVER_CLASSES[protocol].protocol == protocol

    def test_init_minimal(self, protocol: str) -> None:
        """Test that instantiation with minimal parameters work."""
        cls = SERVER_CLASSES[protocol]
        server = cls(host="127.0.0.1", port=8765)
        assert isinstance(server, cls)
        assert server.host == "127.0.0.1"
        assert server.port == 8765
        assert server.handler.__class__.__name__ == "MessagesHandler"

    def test_init_ssl(self, protocol: str, ssl_cert: Dict[str, str]) -> None:
        """Test that instantiation with optional SSL parameters works."""
        cls = SERVER_CLASSES[protocol]
        server = cls(
            host="127.0.0.1",
            port=8765,
            certificate=ssl_cert["server_cert"],
            private_key=ssl_cert["server_pkey"],
        )
        assert getattr(server, "_ssl") is not None

    def test_init_ssl_fails(self, protocol: str) -> None:
        """Test that instantiation with invalid SSL parameters fails."""
        cls = SERVER_CLASSES[protocol]
        with pytest.raises(ValueError):
            cls(
                host="127.0.0.1",
                port=8765,
                certificate="certificate",
                private_key=None,
            )
        with pytest.raises(ValueError):
            cls(
                host="127.0.0.1",
                port=8765,
                certificate=None,
                private_key="private-key",
            )

    def test_init_logger(self, protocol: str) -> None:
        """Test that the 'logger' argument is properly parsed."""
        cls = SERVER_CLASSES[protocol]
        logger = get_logger(f"{cls.__name__}Test")
        srv = cls("127.0.0.1", 8765, logger=logger)
        assert srv.logger is logger

    def test_uri(self, protocol: str) -> None:
        """Test that the `uri` property can properly be accessed."""
        cls = SERVER_CLASSES[protocol]
        srv = cls("127.0.0.1", 8765)
        assert isinstance(srv.uri, str)

    def test_client_names(self, protocol: str) -> None:
        """Test that the `client_names` propety can properly be accessed."""
        cls = SERVER_CLASSES[protocol]
        srv = cls("127.0.0.1", 8765)
        assert srv.client_names == set()
        srv.handler.registered_clients[mock.MagicMock()] = "mock"
        assert srv.client_names == {"mock"}


@pytest_asyncio.fixture(name="server")
async def server_fixture(
    protocol: str,
) -> AsyncIterator[NetworkServer]:
    """Fixture to provide with an instantiated and started NetworkServer."""
    server = build_server(
        protocol=protocol,
        host="127.0.0.1",
        port=8765,
    )
    async with server:
        yield server


@pytest.mark.parametrize("protocol", list_available_protocols())
class TestNetworkServerRegister:
    """Unit tests for `NetworkServer` client-registration methods."""

    @pytest.mark.asyncio
    async def test_server_early_request(self, server: NetworkServer) -> None:
        """Test that early 'JoinRequest' are adequately rejected."""
        ctx = mock.MagicMock()
        req = messaging.JoinRequest("mock", {}).to_string()
        rep = await server.handler.handle_message(req, context=ctx)
        assert isinstance(rep, messaging.JoinReply)
        assert not rep.accept
        assert rep.flag == messaging.flags.REGISTRATION_UNSTARTED

    @pytest.mark.asyncio
    async def test_server_await_client(self, server: NetworkServer) -> None:
        """Test 'wait_for_clients' with a single client."""
        clients_info = asyncio.create_task(
            server.wait_for_clients(min_clients=1)
        )
        join_request = messaging.JoinRequest("mock", {})
        server_reply = asyncio.create_task(
            server.handler.handle_message(join_request.to_string(), context=0)
        )
        info = await clients_info
        assert info == {"mock": {}}
        reply = await server_reply
        assert isinstance(reply, messaging.JoinReply)
        assert reply.accept
        assert reply.flag == messaging.flags.REGISTERED_WELCOME

    @pytest.mark.asyncio
    async def test_server_await_timeout(self, server: NetworkServer) -> None:
        """Test 'wait_for_clients' with an expected timeout error."""
        with pytest.raises(RuntimeError):
            await server.wait_for_clients(timeout=1)

    @pytest.mark.asyncio
    async def test_server_await_clients(self, server: NetworkServer) -> None:
        """Test 'wait_for_clients' with a race between many clients.

        Test that the following cases yield expected behaviors:
        - valid join request with a new name
        - valid join request with a duplicated name
        - duplicated valid join request (same context)
        - late join request (third client with only two places)
        """
        # Set up a server waiting routine and join requests' posting.
        clients_info = server.wait_for_clients(
            min_clients=1, max_clients=2, timeout=2
        )
        join_replies = []
        for idx in range(3):
            req = messaging.JoinRequest("mock", {}).to_string()
            ctx = min(idx, 1)  # first and second contexts will be the same
            join_replies.append(server.handler.handle_message(req, ctx))
        # Run the former routines concurrently. Verify server-side results.
        results = await asyncio.gather(clients_info, *join_replies)
        assert results[0] == {"mock": {}, "mock.1": {}}
        # Verify request-wise replies.
        for idx, rep in enumerate(results[1:]):
            assert isinstance(rep, messaging.JoinReply)
            if idx < 2:  # first and third requests will be accepted
                assert rep.accept, idx
                assert rep.flag == messaging.flags.REGISTERED_WELCOME
            elif idx == 2:  # second request is reundant with the first
                assert rep.accept
                assert rep.flag == messaging.flags.REGISTERED_ALREADY
            else:  # fourth request is a third client when only two are exp.
                assert not rep.accept
                assert rep.flag == messaging.flags.REGISTRATION_CLOSED


@pytest.mark.parametrize("protocol", list_available_protocols())
class TestNetworkServerSend:
    """Unit tests for `NetworkServer` message-sending methods."""

    @pytest.mark.asyncio
    async def test_broadcast_message(self, server: NetworkServer) -> None:
        """Test 'broadcast_message' to all clients.

        Mock the message-sending backend, that has dedicated tests.
        """
        handler = server.handler = mock.create_autospec(server.handler)
        setattr(server.handler, "client_names", {"a", "b", "c"})
        msg = messaging.GenericMessage(action="test", params={})
        await server.broadcast_message(msg)
        assert handler.send_message.await_count == 3
        handler.send_message.assert_has_awaits(
            [mock.call(msg, client, 1, None) for client in ("a", "b", "c")],
            any_order=True,
        )

    @pytest.mark.asyncio
    async def test_broadcast_message_subset(
        self, server: NetworkServer
    ) -> None:
        """Test 'broadcast_message' to a selected subset of clients.

        Mock the message-sending backend, that has dedicated tests.
        """
        handler = server.handler = mock.create_autospec(server.handler)
        msg = messaging.GenericMessage(action="test", params={})
        await server.broadcast_message(msg, clients={"a", "b"})
        assert handler.send_message.await_count == 2
        handler.send_message.assert_has_awaits(
            [mock.call(msg, client, 1, None) for client in ("a", "b")],
            any_order=True,
        )

    @pytest.mark.asyncio
    async def test_send_messages(self, server: NetworkServer) -> None:
        """Test 'send_messages', mocking the message-sending backend.

        Mock the message-sending backend, that has dedicated tests.
        """
        handler = server.handler = mock.create_autospec(server.handler)
        messages = {
            str(i): messaging.GenericMessage(action="test", params={"idx": i})
            for i in range(3)
        }  # type: Dict[str, messaging.Message]
        await server.send_messages(messages)
        assert handler.send_message.await_count == 3
        handler.send_message.assert_has_awaits(
            [mock.call(msg, clt, 1, None) for clt, msg in messages.items()],
            any_order=True,
        )

    @pytest.mark.asyncio
    async def test_send_messages_error(self, server: NetworkServer) -> None:
        """Test 'send_messages' error-raising, mocking the backend."""
        handler = server.handler = mock.create_autospec(server.handler)
        handler.send_message.side_effect = RuntimeError
        messages = {
            str(i): messaging.GenericMessage(action="test", params={"idx": i})
            for i in range(3)
        }  # type: Dict[str, messaging.Message]
        with pytest.raises(RuntimeError):
            await server.send_messages(messages)

    @pytest.mark.asyncio
    async def test_send_message(self, server: NetworkServer) -> None:
        """Test 'send_message' - enabling to mock it elsewhere."""
        server.handler.registered_clients = {0: "mock.0", 1: "mock.1"}
        msg = messaging.GenericMessage(action="test", params={})
        # Create tasks to send a message and let the client collect it.
        req = messaging.GetMessageRequest().to_string()
        send = server.send_message(msg, client="mock.0")
        recv = server.handler.handle_message(req, 0)
        # Check that the send routine works, as does the collection one.
        outpt, reply = await asyncio.gather(send, recv)
        assert outpt is None
        assert reply == msg

    @pytest.mark.asyncio
    async def test_send_message_errors(self, server: NetworkServer) -> None:
        """Test that 'send_message' raises expected exceptions."""
        server.handler.registered_clients = {0: "mock.0", 1: "mock.1"}
        msg = messaging.GenericMessage(action="test", params={})
        # Test case when sending to an unknown client.
        with pytest.raises(KeyError):
            await server.send_message(msg, client="unknown")
        # Test case when sending results with a timeout.
        with pytest.raises(asyncio.TimeoutError):
            await server.send_message(msg, client="mock.0", timeout=1)

    @pytest.mark.asyncio
    async def test_reject_msg_request(self, server: NetworkServer) -> None:
        """Test that 'send_message' properly handles clients' identity."""
        server.handler.registered_clients = {0: "mock.0", 1: "mock.1"}
        msg = messaging.GenericMessage(action="test", params={})
        # Create tasks to send a message and have another client request one.
        req = messaging.GetMessageRequest(timeout=1).to_string()
        send = server.send_message(msg, client="mock.0", timeout=1)
        recv = server.handler.handle_message(req, 1)
        # Check that both routines time out.
        excpt, reply = await asyncio.gather(send, recv, return_exceptions=True)
        assert isinstance(excpt, asyncio.TimeoutError)
        assert isinstance(reply, messaging.Error)
        assert reply.message == messaging.flags.CHECK_MESSAGE_TIMEOUT


@pytest.mark.parametrize("protocol", list_available_protocols())
class TestNetworkServerRecv:
    """Unit tests for `NetworkServer` message-receiving methods."""

    @pytest.mark.asyncio
    async def test_wait_for_messages(self, server: NetworkServer) -> None:
        """Test that 'wait_for_messages' works correctly."""
        server.handler.registered_clients = {0: "mock.0", 1: "mock.1"}
        msg = messaging.GenericMessage(action="test", params={})
        # Create tasks to wait for messages, and receive them.
        wait = server.wait_for_messages()
        recv_0 = server.handler.handle_message(msg.to_string(), context=0)
        recv_1 = server.handler.handle_message(msg.to_string(), context=1)
        # Await all tasks and assert that results match expectations.
        outp, reply_0, reply_1 = await asyncio.gather(wait, recv_0, recv_1)
        assert outp == {"mock.0": msg, "mock.1": msg}
        assert isinstance(reply_0, messaging.Empty)
        assert isinstance(reply_1, messaging.Empty)

    @pytest.mark.asyncio
    async def test_wait_for_messages_subset(
        self, server: NetworkServer
    ) -> None:
        """Test 'wait_for_messages' for a subset of all clients."""
        server.handler.registered_clients = {0: "mock.0", 1: "mock.1"}
        msg = messaging.GenericMessage(action="test", params={})
        # Create tasks to wait for messages of the 1st client and receive it.
        wait = server.wait_for_messages(clients={"mock.1"})
        recv = server.handler.handle_message(msg.to_string(), context=1)
        # Await all tasks and assert that results match expectations.
        outp, reply = await asyncio.gather(wait, recv)
        assert outp == {"mock.1": msg}
        assert isinstance(reply, messaging.Empty)

    @pytest.mark.asyncio
    async def test_wait_for_messages_errors(
        self, server: NetworkServer
    ) -> None:
        """Test that 'wait_for_messages' raises expected errors."""
        server.handler.registered_clients = {0: "mock.0", 1: "mock.1"}
        # Specify to wait for messages from an unknown client.
        with pytest.raises(KeyError):
            await server.wait_for_messages(clients={"unknown"})
        # Wait for a message that never comes, with a timeout.
        with pytest.raises(asyncio.TimeoutError):
            await server.wait_for_messages(timeout=1)

    @pytest.mark.asyncio
    async def test_reject_send_request(self, server: NetworkServer) -> None:
        """Test that 'wait_for_messages' properly handles clients' identity."""
        server.handler.registered_clients = {0: "mock.0", 1: "mock.1"}
        msg_0 = messaging.GenericMessage(action="test-0", params={})
        msg_1 = messaging.GenericMessage(action="test-1", params={})
        # Create tasks to wait for messages of the 1st client and receive it.
        wait_0 = server.wait_for_messages(clients={"mock.0"})
        wait_1 = server.wait_for_messages(clients={"mock.1"})
        recv_1 = server.handler.handle_message(msg_1.to_string(), context=1)
        recv_0 = server.handler.handle_message(msg_0.to_string(), context=0)
        # Await all tasks and assert that results match expectations.
        outp_0, outp_1, reply_1, reply_0 = await asyncio.gather(
            wait_0, wait_1, recv_1, recv_0
        )
        assert outp_0 == {"mock.0": msg_0}
        assert outp_1 == {"mock.1": msg_1}
        assert isinstance(reply_0, messaging.Empty)
        assert isinstance(reply_1, messaging.Empty)
