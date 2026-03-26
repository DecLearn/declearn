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

from typing import List, Tuple, Union

import numpy as np

from declearn.utils.serialize._base import add_serialization_support

__all__ = [
    "unpack_numpy",
    "pack_numpy",
]


def pack_numpy(
    array: np.ndarray, allow_bin: bool = False
) -> Tuple[Union[str, bytes], str, List[int]]:
    """Serialize a numpy array to (data, dtype, shape).

    Parameters
    ----------
    array : np.ndarray
    allow_bin : bool
        If True, use raw bytes. Otherwise, use hex string.

    Returns
    -------
    (data, dtype, shape)
    """
    raw = array.tobytes()
    data = raw if allow_bin else raw.hex()
    return (data, array.dtype.char, list(array.shape))


def unpack_numpy(
    data: Tuple[Union[str, bytes], str, List[int]], allow_bin: bool = False
) -> np.ndarray:
    """Deserialize (data, dtype, shape) into a numpy array.

    Parameters
    ----------
    data : tuple
    allow_bin : bool
        If True, interpret data as raw bytes. Otherwise, as hex string.

    Returns
    -------
    np.ndarray
    """
    dump: Union[str, bytes] = data[0]
    buffer = dump if allow_bin else bytes.fromhex(dump)  # type: ignore
    array = np.frombuffer(buffer, dtype=data[1])
    return array.reshape(data[2]).copy()


add_serialization_support(
    np.ndarray,
    "json",
    lambda a: pack_numpy(a, allow_bin=False),
    lambda d: unpack_numpy(d, allow_bin=False),
    "np.ndarray",
)
add_serialization_support(
    np.ndarray,
    "msgpack",
    lambda a: pack_numpy(a, allow_bin=True),
    lambda d: unpack_numpy(d, allow_bin=True),
    "np.ndarray",
)
