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

"""Unit tests for message-type-verification utils."""

import dataclasses
from unittest import mock

import pytest

from declearn.communication.api import NetworkClient, NetworkServer
from declearn.communication.utils import (
    ErrorMessageException,
    MessageTypeException,
    verify_client_messages_validity,
    verify_server_message_validity,
)
from declearn.messaging import Error, Message, SerializedMessage


@dataclasses.dataclass
class SimpleMessage(Message, register=False):  # type: ignore[call-arg]
    """Stub Message subclass for this module's unit tests."""

    typekey = "simple"

    content: str


@pytest.mark.asyncio
async def test_verify_client_messages_validity_expected_simple():
    """Test 'verify_client_messages_validity' with valid messages."""
    # Setup simple messages and have the server except them.
    netwk = mock.create_autospec(NetworkServer, instance=True)
    messages = {f"client_{i}": SimpleMessage(f"message_{i}") for i in range(3)}
    received = {
        key: SerializedMessage(type(val), val.to_string().split("\n", 1)[1])
        for key, val in messages.items()
    }
    results = await verify_client_messages_validity(
        netwk=netwk, received=received, expected=SimpleMessage
    )
    # Assert that results match expectations, and no message was sent.
    assert isinstance(results, dict)
    assert results == messages
    netwk.broadcast_message.assert_not_called()


@pytest.mark.asyncio
async def test_verify_client_messages_validity_expected_error():
    """Test 'verify_client_messages_validity' with expected Error messages."""
    # Setup simple messages and have the server except them.
    netwk = mock.create_autospec(NetworkServer, instance=True)
    messages = {f"client_{i}": Error(f"message_{i}") for i in range(3)}
    received = {
        key: SerializedMessage(type(val), val.to_string().split("\n", 1)[1])
        for key, val in messages.items()
    }
    results = await verify_client_messages_validity(
        netwk=netwk, received=received, expected=Error
    )
    # Assert that results match expectations, and no message was sent.
    assert isinstance(results, dict)
    assert results == messages
    netwk.broadcast_message.assert_not_called()


@pytest.mark.asyncio
async def test_verify_client_messages_validity_unexpected_types():
    """Test 'verify_client_messages_validity' with invalid messages."""
    # Setup simple messages, but have the server except Error messages.
    netwk = mock.create_autospec(NetworkServer, instance=True)
    messages = {f"client_{i}": SimpleMessage(f"message_{i}") for i in range(3)}
    received = {
        key: SerializedMessage(type(val), val.to_string().split("\n", 1)[1])
        for key, val in messages.items()
    }
    # Assert that an exception is raised.
    with pytest.raises(MessageTypeException):
        await verify_client_messages_validity(
            netwk=netwk, received=received, expected=Error
        )
    # Assert that an Error message was broadcast to all clients.
    netwk.broadcast_message.assert_awaited_once_with(
        message=Error(mock.ANY), clients=set(received)
    )


@pytest.mark.asyncio
async def test_verify_client_messages_validity_unexpected_error():
    """Test 'verify_client_messages_validity' with 'Error' messages."""
    # Setup simple messages, but have one be an Error.
    netwk = mock.create_autospec(NetworkServer, instance=True)
    messages = {f"client_{i}": SimpleMessage(f"message_{i}") for i in range(2)}
    messages["client_2"] = Error("error_message")
    received = {
        key: SerializedMessage(type(val), val.to_string().split("\n", 1)[1])
        for key, val in messages.items()
    }
    # Assert that an exception is raised.
    with pytest.raises(ErrorMessageException):
        await verify_client_messages_validity(
            netwk=netwk, received=received, expected=SimpleMessage
        )
    # Assert that an Error message was broadcast to non-Error-sending clients.
    netwk.broadcast_message.assert_awaited_once_with(
        message=Error(mock.ANY), clients={"client_0", "client_1"}
    )


@pytest.mark.asyncio
async def test_verify_server_message_validity_expected_simple():
    """Test 'verify_server_message_validity' with a valid message."""
    # Setup a simple message matching client expectations.
    netwk = mock.create_autospec(NetworkClient, instance=True)
    message = SimpleMessage("message")
    received = SerializedMessage(
        SimpleMessage, message.to_string().split("\n", 1)[1]
    )
    result = await verify_server_message_validity(
        netwk=netwk, received=received, expected=SimpleMessage
    )
    # Assert that results match expectations, and no message was sent.
    assert isinstance(result, Message)
    assert result == message
    netwk.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_verify_server_message_validity_expected_error():
    """Test 'verify_server_message_validity' with an expected Error message."""
    # Setup a simple message matching client expectations.
    netwk = mock.create_autospec(NetworkClient, instance=True)
    message = Error("message")
    received = SerializedMessage(Error, message.to_string().split("\n", 1)[1])
    result = await verify_server_message_validity(
        netwk=netwk, received=received, expected=Error
    )
    # Assert that results match expectations, and no message was sent.
    assert isinstance(result, Message)
    assert result == message
    netwk.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_verify_server_message_validity_unexpected_type():
    """Test 'verify_server_message_validity' with an unexpected message."""
    # Setup a simple message, but have the client except an Error one.
    netwk = mock.create_autospec(NetworkClient, instance=True)
    message = SimpleMessage("message")
    received = SerializedMessage(
        SimpleMessage, message.to_string().split("\n", 1)[1]
    )
    # Assert that an exception is raised.
    with pytest.raises(MessageTypeException):
        await verify_server_message_validity(
            netwk=netwk, received=received, expected=Error
        )
    # Assert that an Error was sent to the server.
    netwk.send_message.assert_awaited_once_with(message=Error(mock.ANY))


@pytest.mark.asyncio
async def test_verify_server_message_validity_unexpected_error():
    """Test 'verify_server_message_validity' with an unexpected 'Error'."""
    # Setup an unexpected Error message.
    netwk = mock.create_autospec(NetworkClient, instance=True)
    message = Error("message")
    received = SerializedMessage(Error, message.to_string().split("\n", 1)[1])
    # Assert that an exception is raised.
    with pytest.raises(ErrorMessageException):
        await verify_server_message_validity(
            netwk=netwk, received=received, expected=SimpleMessage
        )
    # Assert that no Error was sent to the server.
    netwk.send_message.assert_not_called()
