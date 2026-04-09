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

"""Base API to define messages for DecLearn processes."""

import dataclasses
from abc import ABCMeta
from typing import Any, ClassVar, Dict, Generic, Self, Tuple, Type, TypeVar

import msgpack  # type: ignore

from declearn.utils import (
    access_registered,
    create_types_registry,
    register_from_attr,
)
from declearn.utils.serialize import (
    msgpack_deserialize,
    msgpack_serialize,
)

__all__ = [
    "Message",
]


@create_types_registry(name="Message")
@dataclasses.dataclass
class Message(metaclass=ABCMeta):
    """Abstract base dataclass to define parsable messages.

    A 'Message' is merely an arbitrary data structure that implements
    conversion to and from a serializable dict, and is associated
    with a unique `typekey` string class attribute under which it is
    type-registered.

    All subclasses must be decorated into a `dataclasses.dataclass`.

    Subclasses that only have serializable fields do not need to define
    anything else than `typekey`. If type-conversion is required, they
    may overload the `to_kwargs` method and `from_kwargs` classmethod.

    Subclasses are type-registered by default. This can be prevented
    (e.g. in testing contexts, or when defining an abstract subclass)
    by passing the `register=False` keyword argument at inheritance;
    e.g. `class MyMsg(Message, register=False):`.
    """

    typekey: ClassVar[str]

    def __init_subclass__(
        cls,
        register: bool = True,
    ) -> None:
        """Automatically type-register subclasses."""
        if register:
            register_from_attr(cls, "typekey", group="Message")

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a serializable dict representation of this message."""
        return dataclasses.asdict(self)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        """Parse the message from deserialized attributes."""
        return cls(**kwargs)

    def serialize(self) -> bytes:
        """Convert the message to MessagePack-serialized bytes.

        A header is added to the MessagePack payload:
        - one byte to encode the length of the binary-encoded typekey
        - the binary-encoded typekey

        This header will allow to retrieve the Message's typekey value without
        the need to fully deserialize the Message. Which is useful in our
        deserialization control system.
        """
        data = self.to_kwargs()
        payload = msgpack_serialize(data)
        typekey_bytes = self.typekey.encode("utf-8")
        len_tk_byte = len(typekey_bytes).to_bytes(1, "big")
        return b"".join((len_tk_byte, typekey_bytes, payload))

    @staticmethod
    def parse_typekey_header(bin_msg: bytes) -> Tuple[str, int]:
        """Parse the typekey header of a binary message.

        Also return the index at which the payload starts in the binary data.

        Parameters
        ----------
        bin_msg:
            Binary message with format:
            [1-byte length][typekey string][payload].

        Returns
        -------
        Tuple[str, int]
            - `typekey`: message type identifier.
            - `payload_start`: payload starting index.
        """
        typekey_len = bin_msg[0]
        payload_start = 1 + typekey_len
        typekey = bin_msg[1:payload_start].decode("utf-8")
        return typekey, payload_start


MessageT = TypeVar("MessageT", bound=Message)


class SerializedMessage(Generic[MessageT]):
    """Container for serialized Message instances.

    This class provides an intermediate structure to wrap received
    serialized messages, that enables parsing their exact type and
    therefore running any kind of filtering or validation prior to
    actually de-serializing the message's content (which may cause
    non-trivial time and memory usage, assignment of data on a GPU,
    etc.).

    Attributes
    ----------
    message_cls: Type[MessageT]
        Type of the wrapped message, must be a `Message` subclass.
    bin_data: bytes
        Binary data containing the serialized message (header and payload).
    payload_start: int
        Index at which the payload starts in the binary data.
        Used to efficiently deserialize the payload directly from `bin_data`.

    Usage
    -----
    ```
    >>> proto = SerializedMessage.from_bin_message(bin_msg)
    >>> assert issubclass(proto.message_cls, ExpectedMessageType)
    >>> message = proto.deserialize()  # type: `proto.message_cls`
    ```
    """

    def __init__(
        self,
        message_cls: Type[MessageT],
        bin_data: bytes,
        payload_start: int,
    ) -> None:
        """Instantiate the serialized message container."""
        self.message_cls = message_cls
        self.bin_data = bin_data
        self.payload_start = payload_start

    @property
    def typekey(self) -> str:
        """Typekey string associated with this message."""
        return self.message_cls.typekey

    def deserialize(
        self,
    ) -> MessageT:
        """Deserialize this message into a 'self.message_cls' instance."""
        try:
            # Use a memoryview to avoid allocating byte copies when extracting
            # the payload.
            view = memoryview(self.bin_data)
            data = msgpack_deserialize(view[self.payload_start :])
        except (msgpack.UnpackException, msgpack.ExtraData, TypeError) as exc:
            raise ValueError(
                f"Failed to decode MessagePack dump of '{self.message_cls}' "
                "message."
            ) from exc
        return self.message_cls.from_kwargs(**data)

    @classmethod
    def from_bin_message(
        cls,
        bin_msg: bytes,
    ) -> Self:
        """Parse a binary-serialized message into a `SerializedMessage`."""
        try:
            typekey, payload_start = Message.parse_typekey_header(bin_msg)
        except (IndexError, UnicodeDecodeError) as exc:
            raise TypeError(
                "Input string appears not to be a Message dump."
            ) from exc
        try:
            message_cls = access_registered(typekey, group="Message")
        except KeyError as exc:
            raise KeyError(
                f"No registered Message type matches typekey '{typekey}'."
            ) from exc
        if not issubclass(message_cls, Message):  # pragma: no cover
            raise RuntimeError(
                f"Retrieved a non-Message class '{message_cls}' from 'Message'"
                " type registry. This indicates undue tempering."
            )
        return cls(
            message_cls=message_cls,  # type: ignore
            bin_data=bin_msg,
            payload_start=payload_start,
        )
