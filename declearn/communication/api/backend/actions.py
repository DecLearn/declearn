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

"""Fundamental backend hard-coded message containers for DecLearn.

These classes (and their root ancestor `ActionMessage`) provide with
basic structures to pass information across network communication.

They are designed to be used in the backend of API-defining classes
(namely, `NetworkServer`, `NetworkClient` and the `MessageHandler`
backend utility), and not to be used by end-users (save maybe for
end-users that would write custom communication endpoints, but even
these should in general not have to overload shared backend code).

As for application-side messages, they are left to be designed at
another place (`declearn.messaging`), and to be (de)serialized at
other points of the application, leaving network communications
with the mere job to transmit strings across the network.
"""

import abc
import dataclasses
import json
from typing import Optional

__all__ = [
    "Accept",
    "ActionMessage",
    "Drop",
    "Join",
    "Ping",
    "Recv",
    "Reject",
    "Send",
    "parse_action_from_string",
]


@dataclasses.dataclass
class ActionMessage(metaclass=abc.ABCMeta):
    """Abstract base class for fundamental messages."""

    def to_string(
        self,
    ) -> str:
        """Serialize this 'ActionMessage' to a string."""
        data = dataclasses.asdict(self)
        data["action"] = self.__class__.__name__.lower()
        return json.dumps(data)


@dataclasses.dataclass
class Accept(ActionMessage):
    """Server action message to accept a client."""

    flag: str


@dataclasses.dataclass
class Drop(ActionMessage):
    """Client action message to disconnect from a server."""

    reason: Optional[str] = None


@dataclasses.dataclass
class Join(ActionMessage):
    """Client action message to request joining a server."""

    name: str
    version: Optional[str] = None


@dataclasses.dataclass
class Ping(ActionMessage):
    """Shared empty action message for ping purposes."""


@dataclasses.dataclass
class Recv(ActionMessage):
    """Client action message to get content from the server."""

    timeout: Optional[int] = None


@dataclasses.dataclass
class Reject(ActionMessage):
    """Server action message to reject a client's message."""

    flag: str


@dataclasses.dataclass
class Send(ActionMessage):
    """Action message to post content to or receive content from the server."""

    content: str


_ACTION_CLASSES = [
    Accept,
    Drop,
    Join,
    Ping,
    Recv,
    Reject,
    Send,
]
ACTION_MESSAGES = {cls.__name__.lower(): cls for cls in _ACTION_CLASSES}


def parse_action_from_string(
    string: str,
) -> ActionMessage:
    """Parse a serialized 'ActionMessage' from a string."""
    try:
        data = json.loads(string)
    except json.JSONDecodeError as exc:
        raise ValueError("Failed to parse 'ActionMessage' string.") from exc
    if "action" not in data:
        raise ValueError(
            "Failed to parse 'ActionMessage' string: no 'action' key."
        )
    cls = ACTION_MESSAGES.get(data["action"], None)
    if cls is None:
        raise KeyError(
            "Failed to parse 'ActionMessage' string: no class matches "
            f"'{data['action']}' key."
        )
    return cls(**data)
