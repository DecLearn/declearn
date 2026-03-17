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

"""Numpy-related declearn utils."""

from typing import List, Tuple

import numpy as np

from declearn.utils._json import add_json_support
from declearn.utils._msgpack import add_msgpack_support

__all__ = [
    "deserialize_numpy_bin",
    "deserialize_numpy_str",
    "serialize_numpy_bin",
    "serialize_numpy_str",
]


# FIXME : doc
def serialize_numpy_str(array: np.ndarray) -> Tuple[str, str, List[int]]:
    """Transform a numpy array into a JSON-serializable tuple.

    Inverse operation of `declearn.utils.deserialize_numpy`.
    """
    return (array.tobytes().hex(), array.dtype.char, list(array.shape))


# FIXME : doc
def deserialize_numpy_str(data: Tuple[str, str, List[int]]) -> np.ndarray:
    """Return a numpy array based on serialized information.

    Inverse operation of `declearn.utils.serialize_numpy`.
    """
    buffer = bytes.fromhex(data[0])
    array = np.frombuffer(buffer, dtype=data[1])
    return array.reshape(data[2]).copy()  # copy makes the array writable


# FIXME : doc
def serialize_numpy_bin(array: np.ndarray) -> Tuple[bytes, str, List[int]]:
    return (array.tobytes(), array.dtype.char, list(array.shape))


# FIXME : doc
def deserialize_numpy_bin(data: Tuple[bytes, str, List[int]]) -> np.ndarray:
    buffer = data[0]
    array = np.frombuffer(buffer, dtype=data[1])
    return array.reshape(data[2]).copy()  # copy makes the array writable


add_json_support(
    np.ndarray, serialize_numpy_str, deserialize_numpy_str, "np.ndarray"
)
add_msgpack_support(
    np.ndarray, serialize_numpy_bin, deserialize_numpy_bin, "np.ndarray"
)
