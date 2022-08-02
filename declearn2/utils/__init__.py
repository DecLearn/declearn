# coding: utf-8

"""Shared utils used across declearn.

The key functionalities implemented here are:

Types-registration
------------------
Tools to map class constructors to (name, group) string tuples.

* access_registered:
    Retrieve a registered type from its name and (opt.) group name.
* access_registration_info:
    Retrieve the name (and opt. group) under which a type is registered.
* create_types_registry:
    Create a new types-registration group, with opt. type constraints.
* register_type:
    Register a type, through functional or class-decorator syntax.


JSON-serialization
------------------
Tools to add support for 3rd-party or custom types in JSON files.

* add_json_support:
    Register a (pack, unpack) pair of functions to use on a given type.
* json_pack:
    Function to use as `default` parameter in `json.dump` to extend it.
* json_unpack:
    Function to use as `object_hook` parameter in `json.load` to extend it.

And examples of pre-registered (de)serialization functions:

* (deserialize_numpy, serialize_numpy):
    Pair of functions to (un)pack a numpy ndarray as JSON-serializable data.
"""

from ._json import (
    add_json_support,
    json_pack,
    json_unpack,
)
from ._numpy import (
    deserialize_numpy,
    serialize_numpy,
)
from ._register import (
    access_registered,
    access_registration_info,
    create_types_registry,
    register_type,
)
