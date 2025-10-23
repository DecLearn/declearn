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

"""Unit tests for 'declearn.communication.api.backend.MessagesHandler'."""

import asyncio
import logging
import time
from unittest import mock

import pytest

from declearn.communication.api.backend import MessagesHandler, flags
from declearn.communication.api.backend.actions import (
    Accept,
    Drop,
    Join,
    LegacyReject,
    Ping,
    Recv,
    Reject,
    Send,
)
from declearn.version import VERSION


@pytest.fixture(name="handler")
def fixture_handler() -> MessagesHandler:
    """Setup a MessagesHandler with a mock logger and a 0.1 heartbeat."""
    logger = mock.create_autospec(logging.Logger)
    return MessagesHandler(logger, heartbeat=0.1)


@pytest.mark.asyncio
class TestMessagesHandler:
    """Unit tests for 'declearn.communication.api.backend.MessagesHandler'."""

    # unit tests namespace; pylint: disable=too-many-public-methods

    async def test_handle_invalid_action(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test that an invalid message is rejected."""
        query = "invalid-action-string"
        reply = await handler.handle_message(query, context=mock.MagicMock())
        assert isinstance(reply, Reject)
        assert reply.flag == flags.INVALID_MESSAGE

    async def test_handle_legacy_message(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test that an invalid message is rejected."""
        query = LegacyReject().to_string()
        reply = await handler.handle_message(query, context=mock.MagicMock())
        assert isinstance(reply, LegacyReject)

    async def test_handle_join_open(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test Join action handling with open registration."""
        handler.open_clients_registration()
        query = Join(name="name", version=VERSION).to_string()
        reply = await handler.handle_message(query, context="context")
        assert isinstance(reply, Accept)
        assert reply.flag == flags.REGISTERED_WELCOME
        assert handler.registered_clients["context"] == "name"

    async def test_handle_join_close(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test Join action handling with close registration."""
        handler.close_clients_registration()
        query = Join(name="name", version=VERSION).to_string()
        reply = await handler.handle_message(query, context="context")
        assert isinstance(reply, Reject)
        assert reply.flag == flags.REGISTRATION_CLOSED
        assert "context" not in handler.registered_clients

    async def test_handle_join_wrong_version(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test Join action handling with declearn version mismatch."""
        handler.open_clients_registration()
        query = Join(name="name", version="mock.version.string").to_string()
        reply = await handler.handle_message(query, context="context")
        assert isinstance(reply, Reject)
        assert reply.flag == flags.REJECT_INCOMPATIBLE_VERSION
        assert "context" not in handler.registered_clients

    async def test_handle_join_redundant(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test Join action handling for a pre-registered client."""
        # Register the client a first time and close registration.
        handler.open_clients_registration()
        query = Join(name="name", version=VERSION).to_string()
        reply = await handler.handle_message(query, context="context")
        assert isinstance(reply, Accept)
        assert reply.flag == flags.REGISTERED_WELCOME
        handler.close_clients_registration()
        # Re-process the same registration request.
        reply = await handler.handle_message(query, context="context")
        assert isinstance(reply, Accept)
        assert reply.flag == flags.REGISTERED_ALREADY

    async def test_handle_unregistered(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test handling of a non-Join action from an unregistered client."""
        handler.close_clients_registration()
        query = Ping().to_string()
        reply = await handler.handle_message(query, "context")
        assert isinstance(reply, Reject)
        assert reply.flag == flags.REJECT_UNREGISTERED

    async def test_handle_unexpected_type(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test handling of a server-reserved action from a client."""
        handler.registered_clients = {"context": "client"}
        query = Reject(flag="stub").to_string()
        reply = await handler.handle_message(query, context="context")
        assert isinstance(reply, Reject)
        assert reply.flag == flags.INVALID_MESSAGE

    async def test_handle_recv(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test handling of a Recv action with a pending message."""
        handler.registered_clients = {"context": "client"}
        handler.outgoing_messages["client"] = "message"
        query = Recv(timeout=1).to_string()
        reply = await handler.handle_message(query, context="context")
        assert isinstance(reply, Send)
        assert reply.content == "message"

    async def test_handle_recv_timeout(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test handling of a Recv action that times out."""
        handler.registered_clients = {"context": "client"}
        query = Recv(timeout=0.2).to_string()
        start = time.time()
        reply = await handler.handle_message(query, context="context")
        delay = time.time() - start
        assert isinstance(reply, Reject)
        assert reply.flag == flags.CHECK_MESSAGE_TIMEOUT
        assert 0.2 <= delay <= 0.2 + handler.heartbeat

    async def test_handle_send(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test handling of a Send action from a registered client."""
        handler.registered_clients = {"context": "client"}
        query = Send(content="message").to_string()
        reply = await handler.handle_message(query, context="context")
        assert isinstance(reply, Ping)
        assert handler.incoming_messages["client"] == "message"

    async def test_handle_send_over_pending(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test handling of a Send action with a pending message."""
        handler.registered_clients = {"context": "client"}
        handler.incoming_messages["client"] = "pending"
        query = Send(content="message").to_string()
        coro = handler.handle_message(query, context="context")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(coro, timeout=0.2)
        assert handler.incoming_messages["client"] == "pending"

    async def test_handle_drop(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test handling of a Drop action from a registered client."""
        handler.registered_clients = {"context": "client"}
        query = Drop().to_string()
        reply = await handler.handle_message(query, context="context")
        assert isinstance(reply, Ping)
        assert "content" not in handler.registered_clients

    async def test_handle_ping(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test handling of a Ping action from a registered client."""
        handler.registered_clients = {"context": "client"}
        query = Ping().to_string()
        reply = await handler.handle_message(query, context="context")
        assert isinstance(reply, Ping)

    async def test_post_message(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test posting a message adressed to a registered client."""
        handler.registered_clients = {"context": "client"}
        handler.post_message("message", "client")
        assert handler.outgoing_messages["client"] == "message"

    async def test_post_message_overwrite(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test posting a message that overwrites another pending one."""
        handler.registered_clients = {"context": "client"}
        handler.outgoing_messages["client"] = "pending"
        handler.post_message("message", "client")
        handler.logger.warning.assert_called_once()  # type: ignore
        assert handler.outgoing_messages["client"] == "message"

    async def test_post_message_invalid_client(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test posting a message to an invalid client name."""
        with pytest.raises(KeyError):
            handler.post_message("message", "client")

    async def test_send_message(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test sending a message that gets collected."""
        handler.registered_clients = {"context": "client"}
        send_outp, recv_reply = await asyncio.gather(
            handler.send_message("message", "client", timeout=1),
            handler.handle_message(Recv(timeout=1).to_string(), "context"),
        )
        assert send_outp is None
        assert isinstance(recv_reply, Send) and recv_reply.content == "message"

    async def test_send_message_timeout(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test sending a message that is not collected before timeout."""
        handler.registered_clients = {"context": "client"}
        with pytest.raises(asyncio.TimeoutError):
            await handler.send_message("message", "client", timeout=0.1)

    async def test_check_message(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test collecting a pending message posted by a client."""
        handler.registered_clients = {"context": "client"}
        handler.incoming_messages["client"] = "message"
        output = handler.check_message("client")
        assert output == "message"
        assert "client" not in handler.incoming_messages

    async def test_check_message_no_message(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test checking for a client's message that is not there."""
        handler.registered_clients = {"context": "client"}
        output = handler.check_message("client")
        assert output is None

    async def test_check_message_invalid_client(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test checking for a client's message with invalid name."""
        with pytest.raises(KeyError):
            handler.check_message("client")

    async def test_recv_message(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test receiving a message that gets posted."""
        handler.registered_clients = {"context": "client"}
        recv_message, send_reply = await asyncio.gather(
            handler.recv_message("client", timeout=1),
            handler.handle_message(Send("message").to_string(), "context"),
        )
        assert recv_message == "message"
        assert isinstance(send_reply, Ping)

    async def test_recv_message_timeout(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test receiving a message that is not posted before timeout."""
        handler.registered_clients = {"context": "client"}
        with pytest.raises(asyncio.TimeoutError):
            await handler.recv_message("client", timeout=0.1)

    async def test_wait_for_clients_single_client(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test awaiting registration from a single client."""
        coro_wait_for_client = handler.wait_for_clients(
            min_clients=1, max_clients=None, timeout=1.0
        )
        coro_register_client = handler.handle_message(
            Join("client", VERSION).to_string(), "context"
        )
        outp_wait, outp_join = await asyncio.gather(
            coro_wait_for_client, coro_register_client
        )
        assert outp_wait is None
        assert isinstance(outp_join, Accept)
        assert handler.client_names == {"client"}

    async def test_wait_for_clients_with_timeout(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test awaiting registration for a given delay."""
        # Have the handler await 0.2 seconds and two clients join.
        coro_wait_for_clients = handler.wait_for_clients(
            min_clients=1, max_clients=None, timeout=0.2
        )
        coro_register_client_a = handler.handle_message(
            Join("client", VERSION).to_string(), "context-a"
        )
        coro_register_client_b = handler.handle_message(
            Join("client", VERSION).to_string(), "context-b"
        )
        start = time.time()
        outp_wait, outp_join_a, outp_join_b = await asyncio.gather(
            coro_wait_for_clients,
            coro_register_client_a,
            coro_register_client_b,
        )
        delay = time.time() - start
        # Verify that the delay was respected and both clients were registered.
        assert 0.2 <= delay <= 0.2 + handler.heartbeat
        assert outp_wait is None
        assert isinstance(outp_join_a, Accept)
        assert isinstance(outp_join_b, Accept)
        assert handler.client_names == {"client", "client.1"}

    async def test_wait_for_clients_not_enough_error(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test awaiting registration from too many clients."""
        # Have the handler await 0.2 seconds for 2 clients but only 1 join.
        coro_wait_for_client = handler.wait_for_clients(
            min_clients=2, max_clients=None, timeout=0.2
        )
        coro_register_client = handler.handle_message(
            Join("client", VERSION).to_string(), "context"
        )
        start = time.time()
        excp_wait, outp_join = await asyncio.gather(
            coro_wait_for_client, coro_register_client, return_exceptions=True
        )
        delay = time.time() - start
        # Verify that the delay was respected and a RuntimeError was raised.
        assert 0.2 <= delay <= 0.2 + handler.heartbeat
        assert isinstance(excp_wait, RuntimeError)  # type: ignore
        # Verify that in spite of initial acceptance, handler was purged.
        assert isinstance(outp_join, Accept)  # type: ignore
        assert not handler.registered_clients

    async def test_wait_for_clients_too_many_error(
        self,
        handler: MessagesHandler,
    ) -> None:
        """Test awaiting registration with too many concurrent requests."""
        # Have the handler await maximum 2 clients, but 3 attempt joining.
        coro_wait_for_clients = handler.wait_for_clients(
            min_clients=1, max_clients=2, timeout=0.2
        )
        coro_register_client_a = handler.handle_message(
            Join("client", VERSION).to_string(), "context-a"
        )
        coro_register_client_b = handler.handle_message(
            Join("client", VERSION).to_string(), "context-b"
        )
        coro_register_client_c = handler.handle_message(
            Join("client", VERSION).to_string(), "context-c"
        )
        start = time.time()
        excp_wait, *join_replies = await asyncio.gather(
            coro_wait_for_clients,
            coro_register_client_a,
            coro_register_client_b,
            coro_register_client_c,
            return_exceptions=True,
        )
        delay = time.time() - start
        # Verify that requests were all accepted due to concurrence.
        assert delay < 0.2
        assert all(
            isinstance(reply, Accept)
            for reply in join_replies  # type: ignore
        )
        # Verify that this resulting in a RuntimeError and purging the handler.
        assert isinstance(excp_wait, RuntimeError)  # type: ignore
        assert not handler.registered_clients
