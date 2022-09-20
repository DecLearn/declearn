# coding: utf-8

"""Shared utils of the declearn code rewrite."""

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


__all__ = [
    'deserialize_numpy',
    'serialize_numpy',
]


def serialize_numpy(
        array: np.ndarray
    ) -> Tuple[str, str, List[int]]:
    """Transform a numpy array into a JSON-serializable tuple.

    Inverse operation of `declearn.utils.deserialize_numpy`.
    """
    return (array.tobytes().hex(), array.dtype.char, list(array.shape))


def deserialize_numpy(
        data: Tuple[str, str, List[int]]
    ) -> np.ndarray:
    """Return a numpy array based on serialized information.

    Inverse operation of `declearn.utils.serialize_numpy`.
    """
    buffer = bytes.fromhex(data[0])
    array = np.frombuffer(buffer, dtype=data[1])
    return array.reshape(data[2])
