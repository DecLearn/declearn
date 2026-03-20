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

import msgpack  # type: ignore
import pytest

from declearn.utils._msgpack import (
    msgpack_pack,
    msgpack_unpack,
    pack_int,
    unpack_int,
)


@pytest.mark.parametrize(
    "x",
    [0, 1000, 2**65, -1000, -(2**65)],
    ids=["0", "1000", "2**65", "-1000", "-2**65"],
)
def test_pack_unpack_int(x: int):
    """Test for pack_int and unpack_int."""
    packed = pack_int(x)
    unpacked = unpack_int(packed)
    assert unpacked == x


@pytest.mark.parametrize(
    "x",
    [0, 1000, 2**65, -1000, -(2**65)],
    ids=["0", "1000", "2**65", "-1000", "-2**65"],
)
def test_support_msgpack_int(x: int):
    """Test that msgpack support for ints (lower and upper than 64 bits) is
    functionnal when calling msgpack.(un)packb with custom pack/unpack
    callbacks.
    """
    packed = msgpack.packb(x, default=msgpack_pack)
    unpacked = msgpack.unpackb(packed, object_hook=msgpack_unpack)
    assert unpacked == x
