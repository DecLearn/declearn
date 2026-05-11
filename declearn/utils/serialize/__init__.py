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

"""Shared (de)serialization utils used across DecLearn.

The tools listed below leverage DecLearn's internal (de)serialization 
registries to extend serialization support to custom types (for a given 
serialization format, e.g. JSON), enabling objects of those types to be
(de)serialized under this format.

Common serialization utils
--------------------------
* [add_serialization_support]\
[declearn.utils.serialize.add_serialization_support]:
    Register or update (de)serialization support for a custom type under a
    given serialization format.

JSON serialization
------------------
* [json_serialize][declearn.utils.serialize.json_serialize]:
    Serialize object to JSON string, using extended types support.
* [json_deserialize][declearn.utils.serialize.json_deserialize]:
    Deserialize JSON string to object, using extended types support.
* [json_dump][declearn.utils.serialize.json_dump]:
    Dump a given object to a JSON file, using extended types support.
* [json_load][declearn.utils.serialize.json_load]:
    Load data from a JSON file, using extended types support.
* [list_json_serializable][declearn.utils.serialize.list_json_serializable]:
    Return all types that have a custom JSON-(de)serialization support
    in DecLearn.

MessagePack serialization
-------------------------
* [msgpack_serialize][declearn.utils.serialize.msgpack_serialize]:
    Serialize object to binary data using MessagePack, using extended
    types support.
* [msgpack_deserialize][declearn.utils.serialize.msgpack_deserialize]:
    Deserialize binary data to object using MessagePack, using extended
    types support.
* [msgpack_dump][declearn.utils.serialize.msgpack_dump]:
    Dump a given object to a MessagePack file, using extended types support.
* [msgpack_load][declearn.utils.serialize.msgpack_load]:
    Load data from a MessagePack file, using extended types support.
* [list_msgpack_serializable]\
[declearn.utils.serialize.list_msgpack_serializable]:
    Return all types that have a custom MessagePack-(de)serialization
    support in DecLearn.
"""

__all__ = [
    "json_deserialize",
    "json_dump",
    "json_load",
    "json_serialize",
    "list_json_serializable",
    "list_msgpack_serializable",
    "msgpack_deserialize",
    "msgpack_dump",
    "msgpack_load",
    "msgpack_serialize",
    "add_serialization_support",
]

from ._base import add_serialization_support
from ._json import (
    json_deserialize,
    json_dump,
    json_load,
    json_serialize,
    list_json_serializable,
)
from ._msgpack import (
    list_msgpack_serializable,
    msgpack_deserialize,
    msgpack_dump,
    msgpack_load,
    msgpack_serialize,
)
