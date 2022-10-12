# coding: utf-8

"""Numpy-related declearn utils."""

from typing import List, Tuple

import numpy as np

from declearn.utils._json import add_json_support


__all__ = [
    "deserialize_numpy",
    "serialize_numpy",
]


def serialize_numpy(array: np.ndarray) -> Tuple[str, str, List[int]]:
    """Transform a numpy array into a JSON-serializable tuple.

    Inverse operation of `declearn.utils.deserialize_numpy`.
    """
    return (array.tobytes().hex(), array.dtype.char, list(array.shape))


def deserialize_numpy(data: Tuple[str, str, List[int]]) -> np.ndarray:
    """Return a numpy array based on serialized information.

    Inverse operation of `declearn.utils.serialize_numpy`.
    """
    buffer = bytes.fromhex(data[0])
    array = np.frombuffer(buffer, dtype=data[1])
    return array.reshape(data[2]).copy()  # copy makes the array writable


add_json_support(np.ndarray, serialize_numpy, deserialize_numpy, "np.ndarray")
