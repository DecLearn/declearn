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

"""Unit tests for `declearn.utils._json` tools."""

import json
import time
import warnings
from typing import Any

import pytest

from declearn.utils import (
    add_json_support,
    json_pack,
    json_unpack,
)


class CustomType:
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


def test_add_json_support() -> None:
    """Unit tests for `add_json_support`.

    Note: this only tests that calls pass or fail as expected,
    not that the associated mechanics perform well. These are
    tested in `test_json_pack` and `test_json_unpack_known`.
    """

    # Declare a second, empty custom type for this test only.
    class OtherType:  # pylint: disable=all
        pass

    # Test that registration does not fail.
    add_json_support(CustomType, pack_custom, unpack_custom, "custom")
    # Test that registering twice (wrt type OR name) fails.
    with pytest.raises(KeyError):
        add_json_support(CustomType, pack_custom, unpack_custom, None)
    with pytest.raises(KeyError):
        add_json_support(OtherType, pack_custom, unpack_custom, "custom")
    # Test that `repl=True` works.
    add_json_support(CustomType, pack_custom, unpack_custom, None, repl=True)
    add_json_support(
        OtherType, pack_custom, unpack_custom, "custom", repl=True
    )


def test_json_pack() -> None:
    """Unit tests for `json_pack` with custom-specified objects."""

    # Define a subtype of CustomType (to ensure it is not supported).
    class SubType(CustomType):  # pylint: disable=all
        pass

    # Test that an object of that type cannot be properly packed.
    obj = SubType()
    with pytest.raises(TypeError):
        json_pack(obj)
    with pytest.raises(TypeError):
        json.dumps(obj, default=json_pack)
    # Add JSON support for the type and test that it can now be packed.
    add_json_support(SubType, pack_custom, unpack_custom, name="subtype")
    expected = {"__type__": "subtype", "dump": pack_custom(obj)}
    assert json_pack(obj) == expected
    assert isinstance(json.dumps(obj, default=json_pack), str)


def test_json_unpack_unknown() -> None:
    """Unit tests for `json_unpack` with un-specified objects."""
    # Declare objects that should pass as-is, with and without warnings.
    obj_warn = {"__type__": str(time.time_ns()), "dump": ["lorem ipsum"]}
    obj_pass = {"foo": "foo", "bar": ["lorem ipsum"]}
    # Test that the expected behavior occurs.
    with pytest.warns(UserWarning):
        assert json_unpack(obj_warn) is obj_warn
    with warnings.catch_warnings():  # i.e. assert no warning
        warnings.simplefilter("error")
        assert json_unpack(obj_pass) is obj_pass
    # Test that the functions works as an object hook for `json.loads`.
    struct = {"warn": obj_warn, "pass": obj_pass}
    string = json.dumps(struct)
    with pytest.warns(UserWarning):
        assert json.loads(string, object_hook=json_unpack) == struct


def test_json_unpack_known() -> None:
    """Unit tests for `json_unpack` with custom-specified objects."""
    # Ensure CustomType has been submitted for JSON support.
    add_json_support(
        CustomType, pack_custom, unpack_custom, "custom", repl=True
    )
    # Test that unpacking is performed as expected.
    obj = CustomType()
    msg = {"__type__": "custom", "dump": pack_custom(obj)}
    assert json_unpack(msg) == obj
    # Test that the functions works as an object hook for `json.loads`.
    string = json.dumps(msg)
    assert json.loads(string, object_hook=json_unpack) == obj


def test_json_utils() -> None:
    """Test the full register-pack-unpack pipeline for CustomType."""
    # Ensure CustomType has been submitted for JSON support.
    add_json_support(
        CustomType, pack_custom, unpack_custom, "custom", repl=True
    )
    # Test that packing works thanks to the generic hook.
    struct = {"lorem": "ipsum", "objects": [CustomType(0), CustomType(1)]}
    with pytest.raises(TypeError):
        json.dumps(struct)
    string = json.dumps(struct, default=json_pack)
    # Test that unpacking works thanks to the generic hook.
    assert json.loads(string) != struct
    assert json.loads(string, object_hook=json_unpack) == struct
