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

"""Utils to handle tf.IndexedSlices as part of tensor-processing operations."""

import functools
import warnings
from typing import Any, Callable, TypeVar

import numpy as np
import tensorflow as tf

__all__ = [
    "add_indexed_slices_support",
]


TensorT = TypeVar("TensorT", tf.Tensor, tf.IndexedSlices)


def apply_func_to_tensor_or_slices(
    first: TensorT,
    other: Any,
    tf_op: Callable[[tf.Tensor, Any], tf.Tensor],
) -> TensorT:
    """Run a function on a pair of tensors, adding support for IndexedSlices.

    The intended use of this function is to add support for IndexedSlices
    to basic tensorflow operators, such as `tf.add` or `tf.multiply`.

    Parameters
    ----------
    first: tf.Tensor or tf.IndexedSlices
        Tensor or IndexedSlices data structure.
    other: tf.Tensor or tf.IndexedSlices or np.ndarray or float or int
        Scalar or data array that needs operating onto `first` via `tf_op`.
    tf_op: function(tf.Tensor, any) -> tf.Tensor
        Function that operates on a tf.Tensor and another value.

    Returns
    -------
    output: tf.Tensor or tf.IndexedSlices
        Result from running `tf_op(first, other)` if first is a tf.Tensor,
        or from re-wrapping `tf_op(first.values, other[.values])` into a
        tf.IndexedSlices structure if first is such a structure. The only
        exception is when operating on IndexedSlices and a full-rank array
        or tensor: then a full-rank output is returned, with a warning.

    Raises
    ------
    TypeError
        If `first` and `other` are two tf.IndexedSlices with different
        shapes or non-zero indices.
        If `first` is a tf.IndexedSlices and `func` failed on its values.
    """

    def _same_indexedslices_shape(a, b):
        return (
            a.dense_shape is not None
            and b.dense_shape is not None
            and a.dense_shape.ndim == b.dense_shape.ndim
            and tf.reduce_all(a.dense_shape == b.dense_shape)
            and a.indices.shape == b.indices.shape
            and tf.reduce_all(a.indices == b.indices)
        )

    slice_inp = isinstance(first, tf.IndexedSlices)
    # Case when combining two IndexedSlices objects.
    if slice_inp and isinstance(other, tf.IndexedSlices):
        if _same_indexedslices_shape(first, other):
            values = tf_op(first.values, other.values)
            return tf.IndexedSlices(values, first.indices, first.dense_shape)  # type: ignore
        raise TypeError(
            f"Cannot apply function {tf_op.__name__} to two IndexedSlices "
            "structures with different shapes or indices."
        )
    # Case when operating into an IndexedSlices object.
    if slice_inp:
        # Case when operating with a dense tensor (or array) of same shape.
        if isinstance(other, (tf.Tensor, np.ndarray)):
            if first.shape == other.shape:
                warnings.warn(
                    f"Applying function {tf_op.__name__} to IndexSlices with "
                    "a full-rank array or tensor results in densifying it.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return tf_op(tf.convert_to_tensor(first), other)  # type: ignore
        # Generic case (including mis-shaped tensor, to raise an error).
        try:
            values = tf_op(first.values, other)
        except Exception as exc:
            raise TypeError(
                f"Failed to apply function {tf_op.__name__} to combine a "
                f"{type(other)} object into an IndexedSlices tensor: {exc}."
            ) from exc
        return tf.IndexedSlices(values, first.indices, first.dense_shape)  # type: ignore
    # All other cases (including right-hand slices that will be converted).
    return tf_op(first, other)  # type: ignore


def add_indexed_slices_support(
    tf_op: Callable[[tf.Tensor, Any], tf.Tensor],
    inplc: bool = False,
) -> Callable[[TensorT, Any], TensorT]:
    """Wrap an input function to overload the handling of tf.IndexedSlices.

    Parameters
    ----------
    tf_op: function(tf.Tensor, [any]) -> tf.Tensor
        Tensor-processing operation that needs wrapping.
    inplc: bool, default=False
        Whether to replace the second argument of `tf_op` with None.
        Use this to transform tensor-processing functions (wich, in
        general, have a `name=None` argument) rather than operations.

    Returns
    -------
    func:
        Tensor-processing operation that wraps `tf_op` but supports and
        preserves tf.IndexedSlices inputs as first (and opt. second)
        argument.
        Note that in the rare case when func(slices, dense) is called,
        the output will be dense, and a RuntimeWarning will be raised.
    """
    func = functools.partial(apply_func_to_tensor_or_slices, tf_op=tf_op)
    if inplc:
        func = functools.partial(func, other=None)
    return functools.wraps(tf_op)(func)
