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


"""Tools for JSON-(de)serialization."""

import json
from typing import Any, Dict, List, Optional, Tuple, Type

from declearn.utils.serialize._base import (
    SerialWrapper,
    _decode,
    _encode,
    _list_serializable,
    add_serialization_support,
)

__all__ = [
    "json_deserialize",
    "json_dump",
    "json_load",
    "json_serialize",
]


# JSON serialization utils
def json_serialize(obj: Any) -> str:
    """Serialize object to JSON string.

    See `declearn.utils.serialize.json_deserialize` for the counterpart method.
    """
    return json.dumps(obj, default=_json_encode)


def json_deserialize(data: str):
    """Deserialize JSON string to object.

    See `json_serialize` for the counterpart method.
    """
    return json.loads(data, object_hook=_json_decode)


def json_dump(
    obj: Any,
    path: str,
    encoding: str = "utf-8",
    indent: Optional[int] = None,
) -> None:
    """Dump a given object to a JSON file, using extended types support.

    See `declearn.utils.serialize.add_serialization_support` to extend the
    behaviour of JSON (de)serialization to non-standard types, that will be
    used as part of this function.

    See `json_load` for the counterpart method.
    """
    with open(path, "w", encoding=encoding) as file:
        json.dump(obj, file, default=_json_encode, indent=indent)


def json_load(
    path: str,
    encoding: str = "utf-8",
) -> Any:
    """Load data from a JSON file, using extended types support.

    See `declearn.utils.serialize.add_serialization_support` to extend the
    behaviour of JSON (de)serialization to non-standard types, that will be
    used as part of this function.

    See `json_dump` for the counterpart method.
    """

    with open(path, "r", encoding=encoding) as file:
        return json.load(file, object_hook=_json_decode)


def _json_encode(obj: Any) -> SerialWrapper:
    return _encode(obj, fmt="json")


def _json_decode(obj: Dict[str, Any]) -> Any:
    return _decode(obj, fmt="json")


def list_json_serializable() -> List[Tuple[str, Type]]:
    """Return all types that have a custom JSON-(de)serialization support
    in DecLearn.

    Note that natively serializable types are not listed.

    Returns
    -------
        Alphabetically sorted list of (name, type) pairs, where `name` is the
        type's identifier in the serialization registry.
    """
    return _list_serializable("json")


# Add JSON support for built-in set objects.
add_serialization_support(
    cls=set,
    fmt="json",
    encode=list,
    decode=set,
    name="set",
)
