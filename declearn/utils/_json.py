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

"""Tools to add support for non-standard types' JSON-(de)serialization."""

from __future__ import annotations

import dataclasses
import json
import warnings
from typing import Any, Callable, Dict, Optional, Type, TypedDict

__all__ = [
    "add_json_support",
    "json_dump",
    "json_load",
    "json_pack",
    "json_unpack",
]


JSON_PACK: Dict[Type[Any], SerializeSpec] = {}
JSON_UNPACK: Dict[str, SerializeSpec] = {}

JsonPack = TypedDict(  # pylint: disable=invalid-name
    "JsonPack", {"__type__": str, "dump": Any}
)


@dataclasses.dataclass
class SerializeSpec:
    """Dataclass to wrap a JSON-(de)serialization scheme for a type."""

    cls: Type[Any]
    name: str
    pack: Callable[[Any], Any]  # cls -> any
    unpack: Callable[[Any], Any]  # any -> cls

    def register(self, repl: bool = False) -> None:
        """Register the wrapped type and (un)packing protocols for use.

        Calling this method ensures that the (un)packing protocols
        are added to the `json_pack` and `json_unpack` hooks which
        declearn makes use of when (de)serializing objects to and
        from JSON, effectively adding support for `self.cls`.

        Note that these hooks are also made public, enabling their
        use as part of users' custom code.
        """
        if not repl:
            if self.cls in JSON_PACK:
                raise KeyError(
                    f"Type '{self.cls}' already has a registered "
                    "JSON (de-)serialization specification."
                )
            if self.name in JSON_UNPACK:
                raise KeyError(
                    f"Name '{self.name}' is already in use for the "
                    "JSON (de-)serialization specification of type "
                    f"'{JSON_UNPACK[self.name].cls}'."
                )
        JSON_PACK[self.cls] = self
        JSON_UNPACK[self.name] = self


def add_json_support(
    cls: Type[Any],
    pack: Callable[[Any], Any],
    unpack: Callable[[Any], Any],
    name: Optional[str] = None,
    repl: bool = False,
) -> None:
    """Add or modify JSON (de)serialization support for a custom type.

    Parameters
    ----------
    cls: type
        Type for which to add (or modify) JSON (de)serialization support.
    pack: func(cls) -> any
        Function used to pack objects of type `cls` into an arbitrary
        JSON-serializable object or structure.
    unpack: func(any) -> cls
        Function used to unpack objects of type `cls` from the object
        or structure ouput by the `pack` function.
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
    spec = SerializeSpec(cls, name, pack, unpack)
    spec.register(repl)


def json_pack(obj: Any) -> JsonPack:
    """Pack an object of non-standard type for JSON serialization.

    This function is designed to be passed as `default` parameter
    to the `json.dumps` function. It provides support for object
    types with custom (un)packing protocols registered as part of
    declearn or using `declearn.utils.add_json_support`.
    """
    spec = JSON_PACK.get(type(obj))
    if spec is None:
        raise TypeError(
            f"Object of type '{type(obj)}' is not JSON-serializable.\n"
            "Consider using `declearn.utils.add_json_support` to make it so."
        )
    return {"__type__": spec.name, "dump": spec.pack(obj)}


def json_unpack(obj: Dict[str, Any]) -> Any:
    """Unpack an object of non-standard type as part of JSON deserialization.

    This function is designed to be passed as `object_hook` parameter
    to the `json.loads` function. It provides support for object
    types with custom (un)packing protocols registered as part of
    declearn or using `declearn.utils.add_json_support`.
    """
    # If 'obj' does not conform to JsonPack format, return it as-is.
    if not isinstance(obj, dict) or (set(obj.keys()) != {"__type__", "dump"}):
        return obj
    # If 'obj' is JsonPack but spec is not found, warn before returning as-is.
    spec = JSON_UNPACK.get(obj["__type__"])
    if spec is None:
        warnings.warn(
            "JSON deserializer received a seemingly-packed object "
            f"of name '{obj['__type__']}', the specifications for "
            "which are unavailable.\nIt was returned as-is.",
            stacklevel=2,
        )
        return obj
    # Otherwise, use the recovered spec to unpack the object.
    return spec.unpack(obj["dump"])


def json_dump(
    obj: Any,
    path: str,
    encoding: str = "utf-8",
    indent: Optional[int] = None,
) -> None:
    """Dump a given object to a JSON file, using extended types support.

    This function is merely a shortcut to run the following code:
    ```
    >>> with open(path, "w", encoding=encoding) as file:
    >>>     json.dump(obj, file, default=declearn.utils.json_pack)
    ```

    See `declearn.utils.add_json_support` to extend the behaviour
    of JSON (de)serialization to non-standard types, that will be
    used as part of this function.

    See `declearn.utils.json_load` for the counterpart method.
    """
    with open(path, "w", encoding=encoding) as file:
        json.dump(obj, file, default=json_pack, indent=indent)


def json_load(
    path: str,
    encoding: str = "utf-8",
) -> Any:
    """Load data from a JSON file, using extended types support.

    This function is merely a shortcut to run the following code:
    ```
    >>> with open(path, "r", encoding=encoding) as file:
    >>>     return json.load(file, object_hook=declearn.utils.json_unpack)
    ```

    See `declearn.utils.add_json_support` to extend the behaviour
    of JSON (de)serialization to non-standard types, that will be
    used as part of this function.

    See `declearn.utils.json_dump` for the counterpart method.
    """
    with open(path, "r", encoding=encoding) as file:
        return json.load(file, object_hook=json_unpack)


# Add JSON support for built-in set objects.
add_json_support(
    cls=set,
    pack=list,
    unpack=set,
    name="set",
)
