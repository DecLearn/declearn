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

"""Miscellaneous private backend utils used in model code."""

from typing import List, Set, Tuple

import numpy as np

__all__ = [
    "flatten_numpy_arrays",
    "raise_on_stringsets_mismatch",
    "unflatten_numpy_arrays",
]


def raise_on_stringsets_mismatch(
    received: Set[str],
    expected: Set[str],
    context: str = "expected",
) -> None:
    """Raise a verbose KeyError if two sets of strings do not match.

    Parameters
    ----------
    received: set[str]
        Received set of string values.
    expected: set[str]
        Expected set of string values.
    context: str, default="expected"
        String piece used in the raised exception's description to
        designate the `expected` names.

    Raises
    ------
    KeyError
        In case `received != expected`.
        Verbose about the missing and/or unexpected `received` keys.
    """
    if received != expected:
        missing = expected.difference(received)
        unexpct = received.difference(expected)
        raise KeyError(
            f"Mismatch between input and {context} names:\n"
            + f"Missing key(s) in inputs: {missing}\n" * bool(missing)
            + f"Unexpected key(s) in inputs: {unexpct}\n" * bool(unexpct)
        )


def flatten_numpy_arrays(
    arrays: List[np.ndarray],
) -> List[float]:
    """Flatten a list of numpy arrays into a list of float values.

    Parameters
    ----------
    arrays:
        List of numpy arrays to flatten and concatenate.

    Returns
    -------
    values:
        List of float values made from concatenating, flattening
        and converting input numpy arrays to python float values.
    """
    return [  # type: ignore
        value
        for array in arrays
        for value in array.ravel().astype(float).tolist()
    ]


def unflatten_numpy_arrays(
    values: List[float],
    shapes: List[Tuple[int, ...]],
    dtypes: List[str],
) -> List[np.ndarray]:
    """Unflatten a list of numpy arrays from a list of float values.

    Parameters
    ----------
    values:
        List of float values to put back into a list of numpy arrays.
    shapes:
        List of shapes of the numpy arrays to reconstruct.
    dtypes:
        List of dtypes of the numpy arrays to reconstruct.

    Returns
    -------
    arrays:
        List of numpy arrays storing the input values, enforcing the
        input specs as to shapes and dtypes.
    """
    arrays: List[np.ndarray] = []
    start = 0
    for shape, dtype in zip(shapes, dtypes, strict=False):
        end = start + int(np.prod(shape))
        array = np.array(values[start:end]).astype(dtype).reshape(shape)
        arrays.append(array)
        start = end
    return arrays
