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
    "unpack_numpy_bin",
    "unpack_numpy_str",
    "pack_numpy_bin",
    "pack_numpy_str",
]


def pack_numpy_str(array: np.ndarray) -> Tuple[str, str, List[int]]:
    """Transform a numpy array into a serializable (hex, dtype, shape) tuple.

    The array data is encoded as a hexadecimal string, making it compatible
    with text-based serialization formats such as JSON.

    Parameters
    ----------
    array:
        NumPy array to serialize.

    Returns
    -------
    A tuple (hex_data, dtype, shape) where:
        - hex_data:  array bytes encoded as a hex string.
        - dtype: single-character dtype code (e.g. 'f', 'd', 'i').
        - shape: list of dimension sizes.

    See also
    --------
    `declearn.utils.unpack_numpy_str`: inverse operation.
    """
    return (array.tobytes().hex(), array.dtype.char, list(array.shape))


def unpack_numpy_str(data: Tuple[str, str, List[int]]) -> np.ndarray:
    """Transform a serializable (hex, dtype, shape) tuple into a numpy array.

    Parameters
    ----------
    data:
        A tuple (hex_data, dtype, shape) where:
            - hex_data:  array bytes encoded as a hex string.
            - dtype: single-character dtype code (e.g. 'f', 'd', 'i').
            - shape: list of dimension sizes.

    Returns
    -------
    array:
        Deserialized NumPy array.

    See also
    --------
    `declearn.utils.pack_numpy_str`: inverse operation.
    """
    buffer = bytes.fromhex(data[0])
    array = np.frombuffer(buffer, dtype=data[1])
    return array.reshape(data[2]).copy()  # copy makes the array writable


def pack_numpy_bin(array: np.ndarray) -> Tuple[bytes, str, List[int]]:
    """Transform a numpy array into a serializable (bin, dtype, shape) tuple.

    The array data is stored as raw bytes, making it compatible with
    binary serialization formats such as MessagePack.

    Parameters
    ----------
    array:
        NumPy array to serialize.

    Returns
    -------
    A tuple (bin_data, dtype, shape) where:
        - bin_data:  array raw bytes.
        - dtype: single-character dtype code (e.g. 'f', 'd', 'i').
        - shape: list of dimension sizes.

    See also
    --------
    `declearn.utils.unpack_numpy_bin`: inverse operation.
    """
    return (array.tobytes(), array.dtype.char, list(array.shape))


def unpack_numpy_bin(data: Tuple[bytes, str, List[int]]) -> np.ndarray:
    """Transform a serializable (bin, dtype, shape) tuple into a numpy array.

    Parameters
    ----------
    data:
        A tuple (bin_data, dtype, shape) where:
            - bin_data:  raw array bytes.
            - dtype: single-character dtype code (e.g. 'f', 'd', 'i').
            - shape: list of dimension sizes.

    Returns
    -------
    array:
        Deserialized NumPy array.

    See also
    --------
    `declearn.utils.pack_numpy_bin`: inverse operation.
    """
    buffer = data[0]
    array = np.frombuffer(buffer, dtype=data[1])
    return array.reshape(data[2]).copy()  # copy makes the array writable


add_json_support(np.ndarray, pack_numpy_str, unpack_numpy_str, "np.ndarray")
add_msgpack_support(np.ndarray, pack_numpy_bin, unpack_numpy_bin, "np.ndarray")
