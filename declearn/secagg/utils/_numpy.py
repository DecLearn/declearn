# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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

"""Numpy-related utils for SecAgg, Quantization, etc."""

import functools
from typing import List, Tuple

import numpy as np

__all__ = [
    "get_numpy_float_dtype",
    "get_numpy_uint_dtype",
]


@functools.lru_cache
def get_numpy_uint_dtype(
    int_range: int,
) -> np.dtype:
    """Return the smallest-size numpy uint dtype for a given integer range.

    Parameters
    ----------
    int_range:
        Maximum value defining a positive integer field.
        This value is taken to be included in the field.

    Returns
    -------
    dtype:
        Smallest numpy uint dtype that fits the integer range.

    Raises
    ------
    ValueError
        If `int_range` is too large to fit within a numpy uint dtype.
    """
    # Gather the list of numpy uint types and their bitsize limit.
    uint_types: List[Tuple[np.dtype, int]] = [
        (np.dtype(dtype), np.iinfo(dtype).max.bit_length())
        for dtype in np.unsignedinteger.__subclasses__()
    ]
    # Find the smallest bitsize that can store the target domain values.
    bitsize = int_range.bit_length()
    for dtype, limit in sorted(uint_types, key=lambda x: x[1]):
        if bitsize <= limit:
            return dtype
    # If None, raise a ValueError.
    dtype, limit = uint_types[-1]
    raise ValueError(
        "Cannot quantize values through numpy onto a domain that goes "
        f"beyond the {dtype.name} limit (int_range >= 2**{limit})."
    )


@functools.lru_cache
def get_numpy_float_dtype(
    val_range: float,
) -> np.dtype:
    """Return the smallest-size numpy float dtype for a given float range.

    Parameters
    ----------
    val_range:
        Absolute value defining a range of floating numbers.

    Returns
    -------
    dtype:
        Smallest numpy float dtype that fits the values range.

    Raises
    ------
    ValueError
        If `val_range` is too large to fit within a numpy float dtype.
    """
    # Gather the list of numpy uint types and their bitsize limit.
    float_types: List[Tuple[np.dtype, float]] = [
        (np.dtype(dtype), float(np.finfo(dtype).max))
        for dtype in np.floating.__subclasses__()
    ]
    # Find the smallest bitsize that can store the target domain values.
    for dtype, limit in sorted(float_types, key=lambda x: x[1]):
        if val_range <= limit:
            return dtype
    # If None, raise a ValueError.
    dtype, limit = float_types[-1]
    raise ValueError(
        "Cannot quantize values through numpy onto a domain that goes "
        f"beyond the {dtype.name} limit (val_range >= {limit})."
    )
