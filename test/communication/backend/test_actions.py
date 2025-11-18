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

"""Unit tests for 'declearn.communication.api.backend.actions'."""

import dataclasses
import json

import pytest

from declearn.communication.api.backend import flags
from declearn.communication.api.backend.actions import (
    Accept,
    ActionMessage,
    Drop,
    Join,
    LegacyMessageError,
    LegacyReject,
    Ping,
    Recv,
    Reject,
    Send,
    parse_action_from_string,
)


def assert_action_is_serializable(
    action: ActionMessage,
) -> None:
    """Test that a given 'ActionMessage' is (un)serializable."""
    string = action.to_string()
    assert isinstance(string, str)
    result = parse_action_from_string(string)
    assert isinstance(result, action.__class__)
    assert dataclasses.asdict(result) == dataclasses.asdict(action)


class TestActionMessage:
    """Unit tests for 'ActionMessage' subclasses."""

    def test_accept(self) -> None:
        """Test that 'Accept' is serializable."""
        action = Accept(flag=flags.REGISTERED_WELCOME)
        assert_action_is_serializable(action)

    def test_drop(self) -> None:
        """Test that 'Drop' is serializable."""
        action = Drop()
        assert_action_is_serializable(action)

    def test_join(self) -> None:
        """Test that 'Join' is serializable."""
        action = Join(name="client", version="version")
        assert_action_is_serializable(action)

    def test_ping(self) -> None:
        """Test that 'Ping' is serializable."""
        action = Ping()
        assert_action_is_serializable(action)

    def test_recv(self) -> None:
        """Test that 'Recv' is serializable."""
        action = Recv(timeout=1)
        assert_action_is_serializable(action)

    def test_reject(self) -> None:
        """Test that 'Reject' is serializable."""
        action = Reject(flag=flags.REJECT_UNREGISTERED)
        assert_action_is_serializable(action)

    def test_send(self) -> None:
        """Test that 'Send' is serializable."""
        action = Send(content="stub-content")
        assert_action_is_serializable(action)


class TestParseActionErrors:
    """Unit tests for exception-raising action string parsing."""

    def test_invalid_json(self) -> None:
        """Test that a ValueError is raised on invalid action string."""
        with pytest.raises(ValueError):
            parse_action_from_string("{invalid-json}")

    def test_no_action_key(self) -> None:
        """Test that a ValueError is raised on invalid json dump."""
        string = json.dumps({"data": "stub"})
        with pytest.raises(ValueError):
            parse_action_from_string(string)

    def test_invalid_action_key(self) -> None:
        """Test that a KeyError is raised on invalid action key."""
        string = json.dumps({"action": "stub-action"})
        with pytest.raises(KeyError):
            parse_action_from_string(string)

    def test_legacy_message(self) -> None:
        """Test that a LegacyMessageError is raised on Message dump."""
        string = json.dumps({"typekey": "stub", "data": "stub-data"})
        with pytest.raises(LegacyMessageError):
            parse_action_from_string(string)

    def test_parse_legacy_reject_action(self) -> None:
        """Test that a 'LegacyReject' action cannot be properly parsed."""
        action = LegacyReject()
        string = action.to_string()
        with pytest.raises(LegacyMessageError):
            parse_action_from_string(string)
