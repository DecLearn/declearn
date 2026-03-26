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

"""Common format-agnostic tools for DecLearn object (de)serialization."""

from __future__ import annotations

import dataclasses
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Literal,
    Optional,
    Type,
    TypedDict,
    TypeVar,
)

__all__ = ["add_serialization_support"]

SerialFmt = Literal["msgpack", "json"]  # supported serialization formats

_SERIAL_REGISTRY: Dict[SerialFmt, Dict[Type[Any], SerialSpec]] = {
    "msgpack": {},
    "json": {},
}
"""Dictionary indexed by the serialization format, and then mapping a type to
its serialization spec.
"""

_DESERIAL_REGISTRY: Dict[SerialFmt, Dict[str, SerialSpec]] = {
    "msgpack": {},
    "json": {},
}
"""Dictionary indexed by the serialization format, and then mapping a type
name (key in the registry) to its serialization spec.
"""

T = TypeVar("T")


class SerialWrapper(TypedDict):
    __type__: str
    """Name used to identify the type in the (de)serialization registry."""

    dump: Any
    """Serializable dump of the object."""


@dataclasses.dataclass
class SerialSpec(Generic[T]):
    """Dataclass to wrap a (de)serialization scheme for a type.

    This structure is agnostic of the serialization format.
    """

    cls: Type[T]
    """Object type."""

    name: str
    """Name used to identify the type in the (de)serialization registry."""

    encoder: Callable[[T], Any]  # cls -> any
    """Hook function used to encode objects of this type to a primitive
    serializable representation.
    """

    decoder: Callable[[Any], T]  # any -> cls
    """Hook function used to decode serialized data to build an object
    of the concerned type.
    """


def add_serialization_support(  # noqa: PLR0913
    cls: Type[Any],
    fmt: SerialFmt,
    encode: Callable[[Any], Any],
    decode: Callable[[Any], Any],
    name: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """Add or update a (de)serialization support for a custom type.

    Adding (de)serialization support means registering the function used to
    encode objects of the `cls` class to a `fmt`-serializable structure ; and
    the function used to decode this structure to the original object.

    Parameters
    ----------
    cls: type
        Type for which to add (or overwrite) `fmt`-(de)serialization
        support.
    fmt: SerialFmt
        The serialization format matching the `encode` and `decode` functions.
    encode: func(cls) -> any
        Function used to encode objects of type `cls` into an arbitrary
        `fmt`-serializable object or structure.
    decode: func(any) -> cls
        Function used to decode objects of type `cls` from the object
        or structure output by the `encode` function.
    name: str
        Keyword to use as a marker for serialized instances of type `cls`
        (based on which their deserialization scheme will be retrieved).
        If None, set to `cls.__module__ + '.' + cls.__name__`.
    overwrite: bool
        If True, overwrite any existing serialization spec for `cls` or `name`
        in the registry. Defaults to False.

    Raises
    ------
    KeyError:
        If `overwrite` is False and a conflict is detected in the registry.
    """
    if name is None:
        name = f"{cls.__module__}.{cls.__name__}"

    serial_reg = _SERIAL_REGISTRY[fmt]
    deserial_reg = _DESERIAL_REGISTRY[fmt]
    if not overwrite:
        if cls in serial_reg:
            raise KeyError(
                f"Type '{cls}' already has a registered "
                f"{fmt}-(de)serialization specification."
            )
        if name in deserial_reg:
            raise KeyError(
                f"Name '{name}' is already in use for the "
                f"{fmt}-(de)serialization specification of type "
                f"'{deserial_reg[name].cls}'."
            )
    spec = SerialSpec(cls, name, encode, decode)
    serial_reg[cls] = spec
    deserial_reg[name] = spec


def _encode(obj: Any, fmt: SerialFmt) -> SerialWrapper:
    """Encode an object of non-standard type for serialization in the given
    format.

    This function is designed to be used by an encoding hook in
    a serialization function (e.g. msgpack.packb). It provides support for
    object types with custom (de)coding protocols registered (for the `fmt`
    format) using `declearn.utils.add_serialization_support`.

    Returns
    -------
        `SerialWrapper` dictionary containing the object type name and the
        object `fmt`-serializable dump.
    """
    spec = _SERIAL_REGISTRY[fmt].get(type(obj))
    if spec is None:
        raise TypeError(
            f"Object of type '{type(obj)}' is not {fmt}-serializable.\n"
            "Consider using `declearn.utils.add_serialization_support` to "
            "make it so."
        )
    return {"__type__": spec.name, "dump": spec.encoder(obj)}


def _decode(obj: Dict[str, Any], fmt: SerialFmt) -> Any:
    """Decode an object of non-standard type as part of deserialization from
    the given format.

    This function is designed to be used by a decoding hook in
    a deserialization function (e.g. msgpack.unpackb). It provides support
    for object types with custom (de)coding protocols registered (for the `fmt`
    format) using `declearn.utils.add_serialization_support`.
    """
    # If 'obj' does not conform to SerialWrapper format, return it as-is.
    if not isinstance(obj, dict) or (set(obj.keys()) != {"__type__", "dump"}):
        return obj
    # If 'obj' is SerialWrapper but spec is not found,
    # warn before returning as-is.
    spec = _DESERIAL_REGISTRY[fmt].get(obj["__type__"])
    if spec is None:
        warnings.warn(
            f"{fmt}-deserializer received a seemingly-packed object "
            f"of name '{obj['__type__']}', the specifications for "
            "which are unavailable.\nIt was returned as-is.",
            stacklevel=2,
        )
        return obj
    # Otherwise, use the recovered spec to decode the object.
    return spec.decoder(obj["dump"])
