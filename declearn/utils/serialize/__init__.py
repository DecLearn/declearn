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

"""Shared (de)serialization utils used across declearn."""
# TODO doc

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
