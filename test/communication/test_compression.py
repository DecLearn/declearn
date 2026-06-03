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

"""Unit tests for transport compression-resolving utils."""

import pytest

from declearn.communication.utils._compression import (
    resolve_grpc_compression,
    resolve_ws_compression,
)


@pytest.mark.parametrize("value", [None, "none"], ids=["none", "none-str"])
def test_resolve_ws_compression_no_compression(value):
    """Test that 'no-compression' values resolve to None for websockets."""
    assert resolve_ws_compression(value) is None


def test_resolve_ws_compression_deflate():
    """Test that 'deflate' resolves to the websockets 'deflate' value."""
    assert resolve_ws_compression("deflate") == "deflate"


def test_resolve_ws_compression_invalid():
    """Test that an invalid value raises a ValueError for websockets."""
    with pytest.raises(ValueError):
        resolve_ws_compression("invalid")


@pytest.mark.parametrize("value", [None, "none"], ids=["none", "none-str"])
def test_resolve_grpc_compression_no_compression(value):
    """Test that 'no-compression' values resolve to NoCompression for gRPC."""
    grpc = pytest.importorskip("grpc")
    assert resolve_grpc_compression(value) == grpc.Compression.NoCompression


def test_resolve_grpc_compression_deflate():
    """Test that 'deflate' resolves to grpc.Compression.Deflate."""
    grpc = pytest.importorskip("grpc")
    assert resolve_grpc_compression("deflate") == grpc.Compression.Deflate


def test_resolve_grpc_compression_invalid():
    """Test that an invalid value raises a ValueError for gRPC."""
    with pytest.raises(ValueError):
        resolve_grpc_compression("invalid")
