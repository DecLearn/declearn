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

from declearn.utils._json import add_json_support
from declearn.utils._msgpack import add_msgpack_support

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
    buffer = data[0] if allow_bin else bytes.fromhex(data[0])
    array = np.frombuffer(buffer, dtype=data[1])
    return array.reshape(data[2]).copy()


add_json_support(
    np.ndarray,
    lambda a: pack_numpy(a, allow_bin=False),
    lambda d: unpack_numpy(d, allow_bin=False),
    "np.ndarray",
)
add_msgpack_support(
    np.ndarray,
    lambda a: pack_numpy(a, allow_bin=True),
    lambda d: unpack_numpy(d, allow_bin=True),
    "np.ndarray",
)
