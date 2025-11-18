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

"""Shared utils used across declearn.

The functions and classes exposed by this submodule are listed below,
grouped thematically.

Config serialization
--------------------
Tools to create JSON config dumps of objects and instantiate from them.

* [ObjectConfig][declearn.utils.ObjectConfig]:
    Dataclass to wrap objects' config and interface JSON dumps.
* [deserialize_object][declearn.utils.deserialize_object]:
    Instantiate an object from an ObjectConfig or a JSON file.
* [serialize_object][declearn.utils.serialize_object]:
    Return an ObjectConfig wrapping a given (supported) object.


Types-registration
------------------
Tools to map class constructors to (name, group) string tuples.

* [access_registered][declearn.utils.access_registered]:
    Retrieve a registered type from its name and (opt.) group name.
* [access_registration_info][declearn.utils.access_registration_info]:
    Retrieve the name (and opt. group) under which a type is registered.
* [access_types_mapping][declearn.utils.access_types_mapping]:
    Return a copy of the `{name: type}` mapping of a given group.
* [create_types_registry][declearn.utils.create_types_registry]:
    Create a types group from a base class (as a function or class-decorator).
* [register_type][declearn.utils.register_type]:
    Register a type class (as a function or class-decorator).


JSON-serialization
------------------
Tools to add support for 3rd-party or custom types in JSON files.

* [add_json_support][declearn.utils.add_json_support]:
    Register a (pack, unpack) pair of functions to use on a given type.
* [json_dump][declearn.utils.json_dump]:
    Function to dump data to a JSON file, automatically using `json_pack`.
* [json_load][declearn.utils.json_load]:
    Function to load data from a JSON file, automatically using `json_unpack`.
* [json_pack][declearn.utils.json_pack]:
    Function to use as `default` parameter in `json.dump` to extend it.
* [json_unpack][declearn.utils.json_unpack]:
    Function to use as `object_hook` parameter in `json.load` to extend it.

And examples of pre-registered (de)serialization functions:

* [deserialize_numpy][declearn.utils.deserialize_numpy]
  and [serialize_numpy][declearn.utils.serialize_numpy]:
    Pair of functions to (un)pack a numpy ndarray as JSON-serializable data.

Device-policy utils
-------------------
Utils to access or update parameters defining a global device-selection policy.

* [DevicePolicy][declearn.utils.DevicePolicy]:
    Dataclass to store parameters defining a device-selection policy.
* [get_device_policy][declearn.utils.get_device_policy]:
    Access a copy of the current global device policy.
* [set_device_policy][declearn.utils.set_device_policy]:
    Update the current global device policy.

Logging utils
-------------
Utils to set up and configure loggers:

* [get_logger][declearn.utils.get_logger]:
    Access or create a logger, automating basic handlers' configuration.
* [LOGGING_LEVEL_MAJOR][declearn.utils.LOGGING_LEVEL_MAJOR]:
    Custom "MAJOR" severity level, between stdlib "INFO" and "WARNING".

Miscellaneous
-------------

* [Aggregate][declearn.utils.Aggregate]:
    Abstract base dataclass for cross-peers data aggregation containers.
* [TomlConfig][declearn.utils.TomlConfig]:
    Abstract base class to define TOML-parsable configuration containers.
* [dataclass_from_func][declearn.utils.dataclass_from_func]:
    Automatically build a dataclass matching a function's signature.
* [dataclass_from_init][declearn.utils.dataclass_from_init]:
    Automatically build a dataclass matching a class's init signature.
* [run_as_processes][declearn.utils.run_as_processes]:
    Run coroutines concurrently within individual processes.
"""

from ._aggregate import Aggregate
from ._dataclass import (
    dataclass_from_func,
    dataclass_from_init,
)
from ._device_policy import (
    DevicePolicy,
    get_device_policy,
    set_device_policy,
)
from ._json import (
    add_json_support,
    json_dump,
    json_load,
    json_pack,
    json_unpack,
)
from ._logging import (
    LOGGING_LEVEL_MAJOR,
    get_logger,
)
from ._multiprocess import run_as_processes
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
