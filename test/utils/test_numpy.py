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

"""Unit tests for 'declearn.utils._numpy' tools."""

import numpy as np
import pytest

from declearn.utils import pack_numpy, unpack_numpy
from declearn.utils.serialize import (
    json_deserialize,
    json_serialize,
    msgpack_deserialize,
    msgpack_serialize,
)


@pytest.mark.parametrize("allow_bin", [True, False], ids=["bin", "str"])
def test_pack_unpack_numpy(allow_bin: bool):
    """Test `pack_numpy` and `unpack_numpy` functions."""
    array = np.random.rand(100, 10)
    packed = pack_numpy(array, allow_bin=allow_bin)
    if allow_bin:
        assert isinstance(packed[0], bytes)
    else:
        assert isinstance(packed[0], str)
    array_bis = unpack_numpy(packed, allow_bin=allow_bin)
    assert isinstance(array_bis, type(array))
    assert np.array_equal(array, array_bis)


def test_msgpack_serialization() -> None:
    """Test that MessagePack-serialization of a numpy ndarray works
    properly.
    """
    array = np.random.rand(100, 10)
    dump = msgpack_serialize(array)
    array_bis = msgpack_deserialize(dump)
    assert isinstance(array_bis, type(array))
    assert np.array_equal(array, array_bis)


def test_json_serialization() -> None:
    """Test that JSON-serialization of a numpy ndarray works properly."""
    array = np.random.rand(100, 10)
    dump = json_serialize(array)
    array_bis = json_deserialize(dump)
    assert isinstance(array_bis, type(array))
    assert np.array_equal(array, array_bis)
