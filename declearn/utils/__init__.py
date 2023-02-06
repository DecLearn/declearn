# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""Shared utils used across declearn.

The key functionalities implemented here are:

Config serialization
--------------------
Tools to create JSON config dumps of objects and instantiate from them.

* ObjectConfig:
    Dataclass to wrap objects' config and interface JSON dumps.
* deserialize_object:
    Instantiate an object from an ObjectConfig or a JSON file.
* serialize_object:
    Return an ObjectConfig wrapping a given (supported) object.


Types-registration
------------------
Tools to map class constructors to (name, group) string tuples.

* access_registered:
    Retrieve a registered type from its name and (opt.) group name.
* access_registration_info:
    Retrieve the name (and opt. group) under which a type is registered.
* access_types_mapping:
    Return a copy of the `{name: type}` mapping of a given group.
* create_types_registry:
    Create a types group from a base class (as a function or class-decorator).
* register_type:
    Register a type class (as a function or class-decorator).


JSON-serialization
------------------
Tools to add support for 3rd-party or custom types in JSON files.

* add_json_support:
    Register a (pack, unpack) pair of functions to use on a given type.
* json_dump:
    Function to dump data to a JSON file, automatically using `json_pack`.
* json_load:
    Function to load data from a JSON file, automatically using `json_unpack`.
* json_pack:
    Function to use as `default` parameter in `json.dump` to extend it.
* json_unpack:
    Function to use as `object_hook` parameter in `json.load` to extend it.

And examples of pre-registered (de)serialization functions:

* (deserialize_numpy, serialize_numpy):
    Pair of functions to (un)pack a numpy ndarray as JSON-serializable data.

Miscellaneous
-------------

* TomlConfig:
    Abstract base class to define TOML-parsable configuration containers.
* dataclass_from_func:
    Automatically build a dataclass matching a function's signature.
* dataclass_from_init:
    Automatically build a dataclass matching a class's init signature.
* get_logger:
    Access or create a logger, automating basic handlers' configuration.
"""

from ._dataclass import (
    dataclass_from_func,
    dataclass_from_init,
)
from ._json import (
    add_json_support,
    json_dump,
    json_load,
    json_pack,
    json_unpack,
)
from ._logging import (
    get_logger,
)
from ._numpy import (
    deserialize_numpy,
    serialize_numpy,
)
from ._register import (
    access_registered,
    access_registration_info,
    access_types_mapping,
    create_types_registry,
    register_type,
)
from ._serialize import (
    ObjectConfig,
    deserialize_object,
    serialize_object,
)
from ._toml_config import TomlConfig
