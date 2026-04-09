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

"""Tools for MessagePack-(de)serialization."""

from typing import Any, Dict, List, Tuple, Type, Union

import msgpack  # type: ignore

from declearn.utils.serialize._base import (
    SerialWrapper,
    _decode,
    _encode,
    _list_serializable,
    add_serialization_support,
)

__all__ = [
    "msgpack_deserialize",
    "msgpack_dump",
    "msgpack_load",
    "msgpack_serialize",
]


# MsgPack serialization utils
def msgpack_serialize(obj: Any) -> bytes:
    """Serialize object to binary data using MessagePack, using extended
    types support.

    See `msgpack_deserialize` for the counterpart
    method.
    """
    return msgpack.packb(obj, default=_msgpack_encode)


def msgpack_deserialize(data: Union[bytes, memoryview]):
    """Deserialize binary data to object using MessagePack, using extended
    types support.

    See `msgpack_serialize` for the counterpart
    method.
    """
    return msgpack.unpackb(data, object_hook=_msgpack_decode)


def msgpack_dump(
    obj: Any,
    path: str,
) -> None:
    """Dump a given object to a MessagePack file, using extended types support.

    See `declearn.utils.serialize.add_serialization_support` to extend the
    behaviour of MessagePack (de)serialization to non-standard types, that will
    be used as part of this function.

    See `msgpack_load` for the counterpart method.
    """
    with open(path, "wb") as file:
        msgpack.dump(obj, file, default=_msgpack_encode)


def msgpack_load(
    path: str,
) -> Any:
    """Load data from a MessagePack file, using extended types support.

    See `declearn.utils.serialize.add_serialization_support` to extend the
    behaviour of MessagePack (de)serialization to non-standard types, that will
    be used as part of this function.

    See `msgpack_dump` for the counterpart method.
    """
    with open(path, "rb") as file:
        return msgpack.load(file, object_hook=_msgpack_decode)


def _msgpack_encode(obj: Any) -> SerialWrapper:
    return _encode(obj, fmt="msgpack")


def _msgpack_decode(obj: Dict[str, Any]) -> Any:
    return _decode(obj, fmt="msgpack")


def list_msgpack_serializable() -> List[Tuple[str, Type]]:
    """Return all types that have a custom MessagePack-(de)serialization
    support in DecLearn.

    Note that natively serializable types are not listed.

    Returns
    -------
        Alphabetically sorted list of (name, type) pairs, where `name` is the
        type's identifier in the serialization registry.
    """
    return _list_serializable("msgpack")


# Add MessagePack support for built-in set objects.
add_serialization_support(
    cls=set,
    fmt="msgpack",
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


add_serialization_support(
    cls=int,
    fmt="msgpack",
    encode=pack_int,
    decode=unpack_int,
    name="int",
)
