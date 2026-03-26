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

"""Unit tests for generic and format-specific serialization tools."""

import json
import time
import warnings
from typing import Any

import msgpack  # type: ignore
import pytest

from declearn.utils.serialize._base import SerialFmt, add_serialization_support
from declearn.utils.serialize._json import (
    _json_decode,
    _json_encode,
    json_deserialize,
    json_serialize,
)
from declearn.utils.serialize._msgpack import (
    _msgpack_decode,
    _msgpack_encode,
    msgpack_deserialize,
    msgpack_serialize,
    pack_int,
    unpack_int,
)


class CustomType:  # noqa: PLW1641
    """Mock custom type used for testing purposes."""

    def __init__(self, val: int = 42) -> None:
        """Instantiate the object."""
        self.val = val

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and (self.val == other.val)


def pack_custom(obj: CustomType) -> Any:
    """CustomType-to-serializable function."""
    return [obj.val]


def unpack_custom(dat: Any) -> CustomType:
    """Serializable-to-CustomType function."""
    assert isinstance(dat, list) and (len(dat) == 1)
    assert isinstance(dat[0], int)
    return CustomType(val=dat[0])


# Enable parametrization of the tests by the serialization format using the
# `fmt` fixture.
@pytest.fixture(name="fmt", params=["json", "msgpack"])
def fmt_fixture(request):
    return request.param


class TestSerialization:
    def test_add_serialization_support(self, fmt: SerialFmt) -> None:
        """Unit tests for `add_serialization_support`.

        Note: this only tests that calls pass or fail as expected,
        not that the associated mechanics perform well.
        """

        # Declare a second, empty custom type for this test only.
        class OtherType:  # pylint: disable=all
            pass

        # Test that registration does not fail.
        add_serialization_support(
            CustomType, fmt, pack_custom, unpack_custom, "custom"
        )
        # Test that registering twice (wrt type OR name) fails.
        with pytest.raises(KeyError):
            add_serialization_support(
                CustomType, fmt, pack_custom, unpack_custom, None
            )
        with pytest.raises(KeyError):
            add_serialization_support(
                OtherType, fmt, pack_custom, unpack_custom, "custom"
            )
        # Test that `overwrite=True` works.
        add_serialization_support(
            CustomType,
            fmt,
            pack_custom,
            unpack_custom,
            None,
            overwrite=True,
        )
        add_serialization_support(
            OtherType,
            fmt,
            pack_custom,
            unpack_custom,
            "custom",
            overwrite=True,
        )

    def test_encode(self, fmt: SerialFmt) -> None:
        """Unit tests for `_{fmt}_encode` with custom-specified objects."""
        if fmt == "json":
            encode = _json_encode
        elif fmt == "msgpack":
            encode = _msgpack_encode
        else:
            pytest.fail(f"Unsupported serialization format '{fmt}'")

        # Define a subtype of CustomType (to ensure it is not supported).
        class SubType(CustomType):
            pass

        # Test that an object of that type cannot be properly packed.
        obj = SubType()
        add_serialization_support(
            SubType, fmt, pack_custom, unpack_custom, name="subtype"
        )
        expected = {"__type__": "subtype", "dump": pack_custom(obj)}
        assert encode(obj) == expected

    def test_serialize(self, fmt: SerialFmt) -> None:
        """Unit tests for `{fmt}_serialize` with custom-specified objects."""
        if fmt == "json":
            serialize = json_serialize
            serial_type = str
        elif fmt == "msgpack":
            serialize = msgpack_serialize
            serial_type = bytes
        else:
            pytest.fail(f"Unsupported serialization format '{fmt}'")

        # Define a subtype of CustomType (to ensure it is not supported).
        class SubType(CustomType):
            pass

        # Test that an object of that type cannot be properly packed.
        obj = SubType()
        with pytest.raises(TypeError):
            serialize(obj)
        # Add support for the type and test that it can now be packed.
        add_serialization_support(
            SubType,
            fmt,
            pack_custom,
            unpack_custom,
            name="subtype",
            overwrite=True,
        )
        assert isinstance(serialize(obj), serial_type)

    def test_decode_unknown(self, fmt: SerialFmt) -> None:
        """Unit tests for `_{fmt}_decode` with un-specified objects."""
        if fmt == "json":
            decode = _json_decode
        elif fmt == "msgpack":
            decode = _msgpack_decode
        else:
            pytest.fail(f"Unsupported serialization format '{fmt}'")

        # Declare objects that should pass as-is, with and without warnings.
        obj_warn = {"__type__": str(time.time_ns()), "dump": ["lorem ipsum"]}
        obj_pass = {"foo": "foo", "bar": ["lorem ipsum"]}
        # Test that the expected behavior occurs.
        with pytest.warns(UserWarning):
            assert decode(obj_warn) is obj_warn
        with warnings.catch_warnings():  # i.e. assert no warning
            warnings.simplefilter("error")
            assert decode(obj_pass) is obj_pass

    def test_decode_known(self, fmt: SerialFmt) -> None:
        """Unit tests for `_{fmt}_decode` with custom-specified objects."""
        if fmt == "json":
            decode = _json_decode
        elif fmt == "msgpack":
            decode = _msgpack_decode
        else:
            pytest.fail(f"Unsupported serialization format '{fmt}'")

        # Ensure CustomType has been submitted for support.
        add_serialization_support(
            CustomType,
            fmt,
            pack_custom,
            unpack_custom,
            "custom",
            overwrite=True,
        )
        # Test that unpacking is performed as expected.
        obj = CustomType()
        msg = {"__type__": "custom", "dump": pack_custom(obj)}
        assert decode(msg) == obj

    def test_deserialize_unknown(self, fmt: SerialFmt) -> None:
        """Unit tests for `{fmt}_deserialize` with un-specified objects."""
        # Declare objects that should pass as-is, with and without warnings.
        obj_warn = {"__type__": str(time.time_ns()), "dump": ["lorem ipsum"]}
        obj_pass = {"foo": "foo", "bar": ["lorem ipsum"]}
        # Test that the expected behavior occurs.
        struct = {"warn": obj_warn, "pass": obj_pass}

        if fmt == "json":
            data = json.dumps(struct)
            deserialize = json_deserialize
        elif fmt == "msgpack":
            data = msgpack.packb(struct)
            deserialize = msgpack_deserialize
        else:
            pytest.fail(f"Unsupported serialization format '{fmt}'")

        with pytest.warns(UserWarning):
            assert deserialize(data) == struct

    def test_deserialize_known(self, fmt: SerialFmt) -> None:
        """Unit tests for `{fmt}_deserialize` with custom-specified objects."""
        # Ensure CustomType has been submitted for support.
        add_serialization_support(
            CustomType,
            fmt,
            pack_custom,
            unpack_custom,
            "custom",
            overwrite=True,
        )
        # Test that the deserialization works as expected.
        obj = CustomType()
        msg = {"__type__": "custom", "dump": pack_custom(obj)}

        if fmt == "json":
            data = json.dumps(msg)
            deserialize = json_deserialize
        elif fmt == "msgpack":
            data = msgpack.packb(msg)
            deserialize = msgpack_deserialize
        else:
            pytest.fail(f"Unsupported serialization format '{fmt}'")

        assert deserialize(data) == obj

    def test_serial_deserial(self, fmt: SerialFmt) -> None:
        """Test the full register-serialize-deserialize pipeline for
        CustomType.
        """
        if fmt == "json":
            dumps = json.dumps
            loads = json.loads
            serialize = json_serialize
            deserialize = json_deserialize
        elif fmt == "msgpack":
            dumps = msgpack.packb
            loads = msgpack.unpackb
            serialize = msgpack_serialize
            deserialize = msgpack_deserialize
        else:
            pytest.fail(f"Unsupported serialization format '{fmt}'")

        # Ensure CustomType has been submitted for support.
        add_serialization_support(
            CustomType,
            fmt,
            pack_custom,
            unpack_custom,
            "custom",
            overwrite=True,
        )
        # Test that serialization works.
        struct = {"lorem": "ipsum", "objects": [CustomType(0), CustomType(1)]}
        with pytest.raises(TypeError):
            dumps(struct)
        string = serialize(struct)
        # Test that deserialization works.
        assert loads(string) != struct
        assert deserialize(string) == struct


class TestMsgPackSerialization:
    """Shared unit tests suite for MessagePack-specific serialization utils."""

    @pytest.mark.parametrize(
        "x",
        [0, 1000, 2**65, -1000, -(2**65)],
        ids=["0", "1000", "2**65", "-1000", "-2**65"],
    )
    def test_pack_unpack_int(self, x: int):
        """Test for pack_int and unpack_int."""
        packed = pack_int(x)
        unpacked = unpack_int(packed)
        assert unpacked == x

    @pytest.mark.parametrize(
        "x",
        [0, 1000, 2**65, -1000, -(2**65)],
        ids=["0", "1000", "2**65", "-1000", "-2**65"],
    )
    def test_msgpack_de_serialize_int(self, x: int):
        """Test that msgpack support for ints (lower and upper than 64 bits) is
        functionnal when calling msgpack_(de)serialize.
        """
        packed = msgpack_serialize(x)
        unpacked = msgpack_deserialize(packed)
        assert unpacked == x
