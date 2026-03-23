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

"""Tools to add support for non-standard types'
MessagePack-(de)serialization.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, Dict, Optional, Type, TypedDict

import msgpack  # type: ignore

__all__ = [
    "add_msgpack_support",
    "msgpack_deserialize",
    "msgpack_dump",
    "msgpack_load",
    "msgpack_serialize",
]


REGISTRY: Dict[Type[Any], MsgPackSerializeSpec] = {}
"""Dictionary mapping a type to its serialization spec."""

REVERSE_REGISTRY: Dict[str, MsgPackSerializeSpec] = {}
"""Dictionary mapping a type name/key to its serialization spec."""

MsgPackWrapper = TypedDict("MsgPackWrapper", {"__type__": str, "dump": Any})


# FIXME : refactor with SerializeSpec (json)
@dataclasses.dataclass
class MsgPackSerializeSpec:
    """Dataclass to wrap a MessagePack-(de)serialization scheme for a type."""

    cls: Type[Any]
    name: str
    encoder: Callable[[Any], Any]  # cls -> any
    decoder: Callable[[Any], Any]  # any -> cls

    def register(self, repl: bool = False) -> None:
        """Register the wrapped type and (de)coding protocols for use.

        Calling this method ensures that the (de)coding protocols
        are added to the `encode` and `decode` hooks which
        declearn makes use of when (de)serializing objects to and
        from MessagePack, effectively adding support for `self.cls`.
        """
        if not repl:
            if self.cls in REGISTRY:
                raise KeyError(
                    f"Type '{self.cls}' already has a registered "
                    "MessagePack (de-)serialization specification."
                )
            if self.name in REVERSE_REGISTRY:
                raise KeyError(
                    f"Name '{self.name}' is already in use for the "
                    "MessagePack (de-)serialization specification of type "
                    f"'{REVERSE_REGISTRY[self.name].cls}'."
                )
        REGISTRY[self.cls] = self
        REVERSE_REGISTRY[self.name] = self


def msgpack_serialize(obj: Any) -> bytes:
    """Serialize object to binary data using MessagePack."""
    return msgpack.packb(obj, default=encode)


def msgpack_deserialize(bin_data: bytes):
    """Deserialize binary data to object using MessagePack."""
    return msgpack.unpackb(bin_data, object_hook=decode)


def add_msgpack_support(
    cls: Type[Any],
    encode: Callable[[Any], Any],
    decode: Callable[[Any], Any],
    name: Optional[str] = None,
    repl: bool = False,
) -> None:
    """Add or modify MessagePack (de)serialization support for a custom type.

    Parameters
    ----------
    cls: type
        Type for which to add (or modify) MessagePack (de)serialization
        support.
    encode: func(cls) -> any
        Function used to encode objects of type `cls` into an arbitrary
        MessagePack-serializable object or structure.
    decode: func(any) -> cls
        Function used to decode objects of type `cls` from the object
        or structure output by the `encode` function.
    name: str
        Keyword to use as a marker for serialized instances of type `cls`
        (based on which their deserialization scheme will be retrieved).
        If None, set to `cls.__module__ + '.' + cls.__name__`.
    repl: bool, default=False
        Whether to overwrite any existing specification for type `cls`
        or using name `name`. It is *highly* recommended *not* to set
        this to True unless you know precisely what you are doing.
    """
    if name is None:
        name = f"{cls.__module__}.{cls.__name__}"
    spec = MsgPackSerializeSpec(cls, name, encode, decode)
    spec.register(repl)


def encode(obj: Any) -> MsgPackWrapper:
    """Pack an object of non-standard type for MessagePack serialization.

    This function is designed to be passed as `default` parameter
    to the `msgpack.packb` function. It provides support for object
    types with custom (de)coding protocols registered as part of
    declearn or using `declearn.utils.add_msgpack_support`.
    """
    spec = REGISTRY.get(type(obj))
    if spec is None:
        raise TypeError(
            f"Object of type '{type(obj)}' is not MessagePack-serializable.\n"
            "Consider using `declearn.utils.add_msgpack_support` to make it "
            "so."
        )
    return {"__type__": spec.name, "dump": spec.encoder(obj)}


def decode(obj: Dict[str, Any]) -> Any:
    """Unpack an object of non-standard type as part of MessagePack
    deserialization.

    This function is designed to be passed as `object_hook` parameter
    to the `msgpack.unpackb` function. It provides support for object
    types with custom (de)coding protocols registered as part of
    declearn or using `declearn.utils.add_msgpack_support`.
    """
    # If 'obj' does not conform to JsonPack format, return it as-is.
    if not isinstance(obj, dict) or (set(obj.keys()) != {"__type__", "dump"}):
        return obj
    # If 'obj' is MsgPackWrapper but spec is not found,
    # warn before returning as-is.
    spec = REVERSE_REGISTRY.get(obj["__type__"])
    if spec is None:
        warnings.warn(
            "MessagePack deserializer received a seemingly-packed object "
            f"of name '{obj['__type__']}', the specifications for "
            "which are unavailable.\nIt was returned as-is.",
            stacklevel=2,
        )
        return obj
    # Otherwise, use the recovered spec to unpack the object.
    return spec.decoder(obj["dump"])


def msgpack_dump(
    obj: Any,
    path: str,
) -> None:
    """Dump a given object to a MessagePack file, using extended types support.

    See `declearn.utils.add_msgpack_support` to extend the behaviour
    of MessagePack (de)serialization to non-standard types, that will be
    used as part of this function.

    See `declearn.utils.msgpack_load` for the counterpart method.
    """
    with open(path, "wb") as file:
        msgpack.dump(obj, file, default=encode)


def msgpack_load(
    path: str,
) -> Any:
    """Load data from a MessagePack file, using extended types support.

    See `declearn.utils.add_msgpack_support` to extend the behaviour
    of MessagePack (de)serialization to non-standard types, that will be
    used as part of this function.

    See `declearn.utils.msgpack_dump` for the counterpart method.
    """
    with open(path, "rb") as file:
        return msgpack.load(file, object_hook=decode)


# Add MessagePack support for built-in set objects.
add_msgpack_support(
    cls=set,
    encode=list,
    decode=set,
    name="set",
)


# Add MessagePack support for int, to cover the case of large integers
# (> 64 bits) not natively handled by MessagePack.
# Note: the (un)pack_int methods should logically be called for large int only.
def pack_int(x: int) -> bytes:
    """Serialize an arbitrary-size integer into a minimal-length big-endian
    byte representation using signed two's complement.

    This function ensures that positive integers -whose most significant bit
    would otherwise be interpreted as a sign bit- are encoded with an extra
    leading bit to avoid sign ambiguity.
    """
    if x == 0:
        return b"\x00"

    bits = x.bit_length()
    if x > 0:
        # +1 bit to avoid sign ambiguity
        bits += 1

    length = (bits + 7) // 8
    return x.to_bytes(length, "big", signed=True)


def unpack_int(b: bytes) -> int:
    """Deserialize a big-endian signed byte representation to an arbitrary-size
    integer.
    """
    return int.from_bytes(b, "big", signed=True)


add_msgpack_support(
    cls=int,
    encode=pack_int,
    decode=unpack_int,
    name="int",
)
