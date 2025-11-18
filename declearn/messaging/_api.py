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

"""Base API to define messages for DecLearn processes."""

import dataclasses
import json
from abc import ABCMeta
from typing import Any, ClassVar, Dict, Generic, Self, Type, TypeVar

from declearn.utils import (
    access_registered,
    create_types_registry,
    json_pack,
    json_unpack,
    register_type,
)

__all__ = [
    "Message",
]


@create_types_registry(name="Message")
@dataclasses.dataclass
class Message(metaclass=ABCMeta):
    """Abstract base dataclass to define parsable messages.

    A 'Message' is merely an arbitrary data structure that implements
    conversion to and from a JSON-serializable dict, and is associated
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
            register_type(cls, name=cls.typekey, group="Message")

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this message."""
        # NOTE: override this method to serialize attributes
        #       that are not handled by declearn.utils.json_unpack
        return dataclasses.asdict(self)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        """Parse the message from JSON-deserialized attributes."""
        # NOTE: override this method to de-serialize attributes
        #       that are not handled by declearn.utils.json_pack
        return cls(**kwargs)

    def to_string(self) -> str:
        """Convert the message to a JSON-serialized string."""
        data = self.to_kwargs()
        dump = json.dumps(data, default=json_pack)
        return self.typekey + "\n" + dump


MessageT = TypeVar("MessageT", bound=Message)


class SerializedMessage(Generic[MessageT]):
    """Container for serialized Message instances.

    This class provides an intermediate structure to wrap received
    serialized messages, that enables parsing their exact type and
    therefore running any kind of filtering or validation prior to
    actually de-serializing the message's content (which may cause
    non-trivial time and memory usage, assignment of data on a GPU,
    etc.).

    Usage:
    ```
    >>> proto = SerializedMessage.from_message_string(string)
    >>> assert issubclass(proto.message_cls, ExpectedMessageType)
    >>> message = proto.deserialize()  # type: `proto.message_cls`
    ```
    """

    def __init__(
        self,
        message_cls: Type[MessageT],
        string_data: str,
    ) -> None:
        """Instantiate the serialized message container."""
        self.message_cls = message_cls
        self.string_data = string_data

    @property
    def typekey(self) -> str:
        """Typekey string associated with this message."""
        return self.message_cls.typekey

    def deserialize(
        self,
    ) -> MessageT:
        """Deserialize this message into a 'self.message_cls' instance."""
        try:
            data = json.loads(self.string_data, object_hook=json_unpack)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to decode JSON dump of '{self.message_cls}' message."
            ) from exc
        return self.message_cls.from_kwargs(**data)

    @classmethod
    def from_message_string(
        cls,
        string: str,
    ) -> Self:
        """Parse a serialized message string into a 'SerializedMessage'."""
        try:
            typekey, string_data = string.split("\n", 1)
        except ValueError as exc:
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
            string_data=string_data,
        )
