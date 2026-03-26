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

"""Unit tests for `declearn.utils._json` tools."""

import json
import time
import warnings
from typing import Any

import pytest

from declearn.utils import (
    add_serialization_support,
)
from declearn.utils._json import (
    _json_decode,
    _json_encode,
    json_deserialize,
    json_serialize,
)


# TODO make more generic (any format)
class CustomType:  # noqa: PLW1641
    """Mock custom type used for testing purposes."""

    def __init__(self, val: int = 42) -> None:
        """Instantiate the object."""
        self.val = val

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and (self.val == other.val)


def pack_custom(obj: CustomType) -> Any:
    """CustomType-to-JSON-serializable function."""
    return [obj.val]


def unpack_custom(dat: Any) -> CustomType:
    """JSON-serializable-to-CustomType function."""
    assert isinstance(dat, list) and (len(dat) == 1)
    assert isinstance(dat[0], int)
    return CustomType(val=dat[0])


def test_add_serialization_support() -> None:
    """Unit tests for `add_serialization_support`.

    Note: this only tests that calls pass or fail as expected,
    not that the associated mechanics perform well.
    """

    # Declare a second, empty custom type for this test only.
    class OtherType:  # pylint: disable=all
        pass

    # Test that registration does not fail.
    add_serialization_support(
        CustomType, "json", pack_custom, unpack_custom, "custom"
    )
    # Test that registering twice (wrt type OR name) fails.
    with pytest.raises(KeyError):
        add_serialization_support(
            CustomType, "json", pack_custom, unpack_custom, None
        )
    with pytest.raises(KeyError):
        add_serialization_support(
            OtherType, "json", pack_custom, unpack_custom, "custom"
        )
    # Test that `overwrite=True` works.
    add_serialization_support(
        CustomType, "json", pack_custom, unpack_custom, None, overwrite=True
    )
    add_serialization_support(
        OtherType, "json", pack_custom, unpack_custom, "custom", overwrite=True
    )


def test_json_encode() -> None:
    """Unit tests for `_json_encode` with custom-specified objects."""

    # Define a subtype of CustomType (to ensure it is not supported).
    class SubType(CustomType):
        pass

    # Test that an object of that type cannot be properly packed.
    obj = SubType()
    add_serialization_support(
        SubType, "json", pack_custom, unpack_custom, name="subtype"
    )
    expected = {"__type__": "subtype", "dump": pack_custom(obj)}
    assert _json_encode(obj) == expected


def test_json_serialize() -> None:
    """Unit tests for `json_serialize` with custom-specified objects."""

    # Define a subtype of CustomType (to ensure it is not supported).
    class SubType(CustomType):  # pylint: disable=all
        pass

    # Test that an object of that type cannot be properly packed.
    obj = SubType()
    with pytest.raises(TypeError):
        json_serialize(obj)
    # Add JSON support for the type and test that it can now be packed.
    add_serialization_support(
        SubType,
        "json",
        pack_custom,
        unpack_custom,
        name="subtype",
        overwrite=True,
    )
    assert isinstance(json_serialize(obj), str)


def test_json_decode_unknown() -> None:
    """Unit tests for `_json_decode` with un-specified objects."""
    # Declare objects that should pass as-is, with and without warnings.
    obj_warn = {"__type__": str(time.time_ns()), "dump": ["lorem ipsum"]}
    obj_pass = {"foo": "foo", "bar": ["lorem ipsum"]}
    # Test that the expected behavior occurs.
    with pytest.warns(UserWarning):
        assert _json_decode(obj_warn) is obj_warn
    with warnings.catch_warnings():  # i.e. assert no warning
        warnings.simplefilter("error")
        assert _json_decode(obj_pass) is obj_pass


def test_json_decode_known() -> None:
    """Unit tests for `_json_decode` with custom-specified objects."""
    # Ensure CustomType has been submitted for JSON support.
    add_serialization_support(
        CustomType,
        "json",
        pack_custom,
        unpack_custom,
        "custom",
        overwrite=True,
    )
    # Test that unpacking is performed as expected.
    obj = CustomType()
    msg = {"__type__": "custom", "dump": pack_custom(obj)}
    assert _json_decode(msg) == obj


def test_json_deserialize_known() -> None:
    # Ensure CustomType has been submitted for JSON support.
    add_serialization_support(
        CustomType,
        "json",
        pack_custom,
        unpack_custom,
        "custom",
        overwrite=True,
    )
    # Test that the deserialization works as expected.
    obj = CustomType()
    msg = {"__type__": "custom", "dump": pack_custom(obj)}
    string = json.dumps(msg)
    assert json_deserialize(string) == obj


def test_json_deserialize_unknown() -> None:
    """Unit tests for `json_deserialize` with un-specified objects."""
    # Declare objects that should pass as-is, with and without warnings.
    obj_warn = {"__type__": str(time.time_ns()), "dump": ["lorem ipsum"]}
    obj_pass = {"foo": "foo", "bar": ["lorem ipsum"]}
    # Test that the expected behavior occurs.
    struct = {"warn": obj_warn, "pass": obj_pass}
    string = json.dumps(struct)
    with pytest.warns(UserWarning):
        assert json_deserialize(string) == struct


def test_json_utils() -> None:
    """Test the full register-serialize-deserialize pipeline for CustomType."""
    # Ensure CustomType has been submitted for JSON support.
    add_serialization_support(
        CustomType,
        "json",
        pack_custom,
        unpack_custom,
        "custom",
        overwrite=True,
    )
    # Test that serialization works.
    struct = {"lorem": "ipsum", "objects": [CustomType(0), CustomType(1)]}
    with pytest.raises(TypeError):
        json.dumps(struct)
    string = json_serialize(struct)
    # Test that deserialization works.
    assert json.loads(string) != struct
    assert json_deserialize(string) == struct
