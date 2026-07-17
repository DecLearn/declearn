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

from __future__ import annotations

import abc
import dataclasses
from typing import Optional

import msgpack  # type: ignore

from declearn.utils.serialize import msgpack_deserialize, msgpack_serialize

__all__ = [
    "Accept",
    "ActionMessage",
    "Drop",
    "Join",
    "Ping",
    "Recv",
    "Reject",
    "Send",
]


@dataclasses.dataclass
class ActionMessage(metaclass=abc.ABCMeta):  # noqa: B024
    """Abstract base class for fundamental messages."""

    def serialize(
        self,
    ) -> bytes:
        """Serialize this `ActionMessage` to bytes."""
        data = dataclasses.asdict(self)
        data["action"] = self.__class__.__name__.lower()
        return msgpack_serialize(data)

    @staticmethod
    def deserialize(
        bin_msg: bytes,
    ) -> ActionMessage:
        """Parse a serialized `ActionMessage` from bytes.

        Parameters
        ----------
        bin_msg:
            Serialized `ActionMessage` instance bytes.

        Returns
        -------
        action:
            `ActionMessage` recovered from binary data.

        Raises
        ------
        KeyError
            If the bytes cannot be mapped to an `ActionMessage` class.
        ValueError
            If the bytes cannot be parsed properly.
        """
        try:
            data = msgpack_deserialize(bin_msg)
        except (msgpack.UnpackException, msgpack.ExtraData) as exc:
            raise ValueError("Failed to parse 'ActionMessage' bytes.") from exc
        except TypeError as exc:
            err_msg = "Failed to parse 'ActionMessage'."
            if not isinstance(bin_msg, bytes):
                err_msg += f" Bytes expected, not '{type(bin_msg).__name__}'."
            if isinstance(bin_msg, str):
                err_msg += (
                    " Make sure to use the same DecLearn version everywhere."
                )
            raise ValueError(err_msg) from exc

        if "action" not in data:
            raise ValueError(
                "Failed to parse 'ActionMessage' bytes: no 'action' key."
            )
        action = data.pop("action")
        cls = ACTION_MESSAGES.get(action, None)
        if cls is None:
            raise KeyError(
                "Failed to parse 'ActionMessage' bytes: no class matches "
                f"'{action}' key."
            )
        return cls(**data)


@dataclasses.dataclass
class Accept(ActionMessage):
    """Server action message to accept a client.

    Fields
    ------
    flag:
        String communication flag that specifies status or any information
        coming with this `Accept` message.
    """

    flag: str


@dataclasses.dataclass
class Drop(ActionMessage):
    """Client action message to disconnect from a server.

    Fields
    ------
    reason:
        Optional string that gives the reason behind the drop.
    """

    reason: Optional[str] = None


@dataclasses.dataclass
class Join(ActionMessage):
    """Client action message to request joining a server.

    Fields
    ------
    name:
        Requesting client name.
    version:
        Declearn version used by the requesting client.
    """

    name: str
    version: str


@dataclasses.dataclass
class Ping(ActionMessage):
    """Shared empty action message for ping purposes."""


@dataclasses.dataclass
class Recv(ActionMessage):
    """Client action message to get content from the server.

    Fields
    ------
    timeout:
        Optional timeout value for receiving content from the server.
    """

    timeout: Optional[float] = None


@dataclasses.dataclass
class Reject(ActionMessage):
    """Server action message to reject a client's message.

    Fields
    ------
    flag:
        String communication flag that specifies status or any information
        coming with this `Reject` message.
    """

    flag: str


@dataclasses.dataclass
class Send(ActionMessage):
    """Action message to post content to or receive content from the server.

    Fields
    ------
    content:
        Conveyed binary-serialized content.
    """

    content: bytes


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
