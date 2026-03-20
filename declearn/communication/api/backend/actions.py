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

import msgpack  # type: ignore

from declearn.version import VERSION

# TODO remove deleted methods ?
__all__ = [
    "Accept",
    "ActionMessage",
    "Drop",
    "Join",
    "LegacyReject",
    "LegacyMessageError",
    "Ping",
    "Recv",
    "Reject",
    "Send",
    "parse_action_from_bytes",
]


class LegacyMessageError(Exception):
    """Custom exception to denote legacy Message being received."""


@dataclasses.dataclass
class ActionMessage(metaclass=abc.ABCMeta):  # noqa: B024
    """Abstract base class for fundamental messages."""

    def to_bytes(
        self,
    ) -> bytes:
        """Serialize this 'ActionMessage' to bytes."""
        data = dataclasses.asdict(self)
        data["action"] = self.__class__.__name__.lower()
        return msgpack.packb(data)


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
    version: str


@dataclasses.dataclass
class Ping(ActionMessage):
    """Shared empty action message for ping purposes."""


@dataclasses.dataclass
class Recv(ActionMessage):
    """Client action message to get content from the server."""

    timeout: Optional[float] = None


@dataclasses.dataclass
class Reject(ActionMessage):
    """Server action message to reject a client's message."""

    flag: str


@dataclasses.dataclass
class Send(ActionMessage):
    """Action message to post content to or receive content from the server."""

    content: bytes
    """Conveyed binary-serialized content."""


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


def parse_action_from_bytes(
    bin_data: bytes,
) -> ActionMessage:
    """Parse a serialized `ActionMessage` from bytes.

    Parameters
    ----------
    bin_data:
        Serialized `ActionMessage` instance bytes.

    Returns
    -------
    action:
        `ActionMessage` recovered from `bin_data`.

    Raises
    ------
    KeyError
        If the bytes cannot be mapped to an `ActionMessage` class.
    ValueError
        If the bytes cannot be parsed properly.
    """
    try:
        data = msgpack.unpackb(bin_data)
    except (msgpack.UnpackException, msgpack.ExtraData, TypeError) as exc:
        raise ValueError("Failed to parse 'ActionMessage' bytes.") from exc
    if "action" not in data:
        raise ValueError(
            "Failed to parse 'ActionMessage' bytes: no 'action' key."
        )
    action = data.pop("action")
    cls = ACTION_MESSAGES.get(action, None)
    if cls is None:
        raise KeyError(
            "Failed to parse 'ActionMessage' bytes: no class matches "
            f"'{data['action']}' key."
        )
    return cls(**data)


# TODO : remove when handle version / byte-string incompatibilities
@dataclasses.dataclass
class LegacyReject(ActionMessage):
    """Server action to reject a legacy client's (registration) message.

    This message will be serialized in a way that is compatible with the
    legacy message parser, but not with the current one.
    """

    # TODO : remove ?
    def to_string(
        self,
    ) -> str:
        message = (
            "Cannot communicate due to the DecLearn version in use. "
            f"Please update to `declearn ~= {VERSION}`."
        )
        return json.dumps({"typekey": "error", "message": message})
