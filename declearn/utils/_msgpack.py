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
    "msgpack_dump",
    "msgpack_load",
    "msgpack_pack",
    "msgpack_unpack",
]


PACK_REGISTRY: Dict[Type[Any], MsgPackSerializeSpec] = {}
UNPACK_REGISTRY: Dict[str, MsgPackSerializeSpec] = {}

MsgPackWrapper = TypedDict("MsgPackWrapper", {"__type__": str, "dump": Any})


# FIXME : refactor with SerializeSpec (json)
@dataclasses.dataclass
class MsgPackSerializeSpec:
    """Dataclass to wrap a MessagePack-(de)serialization scheme for a type."""

    cls: Type[Any]
    name: str
    pack: Callable[[Any], Any]  # cls -> any
    unpack: Callable[[Any], Any]  # any -> cls

    def register(self, repl: bool = False) -> None:
        """Register the wrapped type and (un)packing protocols for use.

        Calling this method ensures that the (un)packing protocols
        are added to the `msgpack_pack` and `msgpack_unpack` hooks which
        declearn makes use of when (de)serializing objects to and
        from MessagePack, effectively adding support for `self.cls`.

        Note that these hooks are also made public, enabling their
        use as part of users' custom code.
        """
        if not repl:
            if self.cls in PACK_REGISTRY:
                raise KeyError(
                    f"Type '{self.cls}' already has a registered "
                    "MessagePack (de-)serialization specification."
                )
            if self.name in UNPACK_REGISTRY:
                raise KeyError(
                    f"Name '{self.name}' is already in use for the "
                    "MessagePack (de-)serialization specification of type "
                    f"'{UNPACK_REGISTRY[self.name].cls}'."
                )
        PACK_REGISTRY[self.cls] = self
        UNPACK_REGISTRY[self.name] = self


def add_msgpack_support(
    cls: Type[Any],
    pack: Callable[[Any], Any],
    unpack: Callable[[Any], Any],
    name: Optional[str] = None,
    repl: bool = False,
) -> None:
    """Add or modify MessagePack (de)serialization support for a custom type.

    Parameters
    ----------
    cls: type
        Type for which to add (or modify) MessagePack (de)serialization
        support.
    pack: func(cls) -> any
        Function used to pack objects of type `cls` into an arbitrary
        MessagePack-serializable object or structure.
    unpack: func(any) -> cls
        Function used to unpack objects of type `cls` from the object
        or structure output by the `pack` function.
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
    spec = MsgPackSerializeSpec(cls, name, pack, unpack)
    spec.register(repl)


def msgpack_pack(obj: Any) -> MsgPackWrapper:
    """Pack an object of non-standard type for MessagePack serialization.

    This function is designed to be passed as `default` parameter
    to the `msgpack.packb` function. It provides support for object
    types with custom (un)packing protocols registered as part of
    declearn or using `declearn.utils.add_msgpack_support`.
    """
    spec = PACK_REGISTRY.get(type(obj))
    if spec is None:
        raise TypeError(
            f"Object of type '{type(obj)}' is not MessagePack-serializable.\n"
            "Consider using `declearn.utils.add_msgpack_support` to make it "
            "so."
        )
    return {"__type__": spec.name, "dump": spec.pack(obj)}


def msgpack_unpack(obj: Dict[str, Any]) -> Any:
    """Unpack an object of non-standard type as part of MessagePack
    deserialization.

    This function is designed to be passed as `object_hook` parameter
    to the `msgpack.unpackb` function. It provides support for object
    types with custom (un)packing protocols registered as part of
    declearn or using `declearn.utils.add_msgpack_support`.
    """
    # If 'obj' does not conform to JsonPack format, return it as-is.
    if not isinstance(obj, dict) or (set(obj.keys()) != {"__type__", "dump"}):
        return obj
    # If 'obj' is MsgPackWrapper but spec is not found,
    # warn before returning as-is.
    spec = UNPACK_REGISTRY.get(obj["__type__"])
    if spec is None:
        warnings.warn(
            "MessagePack deserializer received a seemingly-packed object "
            f"of name '{obj['__type__']}', the specifications for "
            "which are unavailable.\nIt was returned as-is.",
            stacklevel=2,
        )
        return obj
    # Otherwise, use the recovered spec to unpack the object.
    return spec.unpack(obj["dump"])


def msgpack_dump(
    obj: Any,
    path: str,
) -> None:
    """Dump a given object to a MessagePack file, using extended types support.

    This function is merely a shortcut to run the following code:
    ```
    >>> with open(path, "wb") as file:
    >>>     msgpack.dump(obj, file, default=declearn.utils.msgpack_pack)
    ```

    See `declearn.utils.add_msgpack_support` to extend the behaviour
    of MessagePack (de)serialization to non-standard types, that will be
    used as part of this function.

    See `declearn.utils.msgpack_load` for the counterpart method.
    """
    with open(path, "wb") as file:
        msgpack.dump(obj, file, default=msgpack_pack)


def msgpack_load(
    path: str,
) -> Any:
    """Load data from a MessagePack file, using extended types support.

    This function is merely a shortcut to run the following code:
    ```
    >>> with open(path, "rb") as file:
    >>>     return msgpack.load(
    >>>         file, object_hook=declearn.utils.msgpack_unpack
    >>>     )
    ```

    See `declearn.utils.add_msgpack_support` to extend the behaviour
    of MessagePack (de)serialization to non-standard types, that will be
    used as part of this function.

    See `declearn.utils.msgpack_dump` for the counterpart method.
    """
    with open(path, "rb") as file:
        return msgpack.load(file, object_hook=msgpack_unpack)


# Add MessagePack support for built-in set objects.
add_msgpack_support(
    cls=set,
    pack=list,
    unpack=set,
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
    pack=pack_int,
    unpack=unpack_int,
    name="int",
)
