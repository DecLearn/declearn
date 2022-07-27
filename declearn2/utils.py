# coding: utf-8

"""Shared utils of the declearn code rewrite."""

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


__all__ = [
    'deserialize_numpy',
    'serialize_numpy',
    'unpack_batch',
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


def unpack_batch(
        batch: Union[ArrayLike, List[Optional[ArrayLike]]],
    ) -> Tuple[ArrayLike, Optional[ArrayLike], Optional[ArrayLike]]:
    """Unpack (inputs, y_true, s_wght) from an input batch.

    `inputs` is a (structure of) input data arrays
    `y_true` may not be specified, and returned as None
    `s_wght` may not be specified, and returned as None
    """
    if not isinstance(batch, (list, tuple)):
        return (batch, None, None)
    if batch and (batch[0] is not None):
        if len(batch) == 1:
            return (batch[0], None, None)
        if len(batch) == 2:
            return (batch[0], batch[1], None)
        if len(batch) == 3:
            return (batch[0], batch[1], batch[2])
    raise TypeError("'batch' should be a list of up to 3 data arrays.")
