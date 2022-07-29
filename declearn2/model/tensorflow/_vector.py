# coding: utf-8

"""TensorflowVector gradients container."""

import json
from typing import Any, Callable, Dict, Optional, Tuple, Union

import tensorflow as tf  # type: ignore
# false-positive; pylint: disable=no-name-in-module
from tensorflow.python.framework.ops import EagerTensor  # type: ignore
# pylint: enable=no-name-in-module

from declearn2.model.api import NumpyVector, Vector, register_vector_type
from declearn2.utils import deserialize_numpy, serialize_numpy


@register_vector_type(tf.Tensor, EagerTensor, tf.IndexedSlices)
class TensorflowVector(Vector):
    """Vector subclass to store tensorflow tensors.

    This Vector is designed to store a collection of named
    TensorFlow tensors, enabling computations that are either
    applied to each and every coefficient, or imply two sets
    of aligned coefficients (i.e. two TensorflowVector with
    similar specifications).

    Note that support for IndexedSlices is implemented,
    as these are a common type for auto-differentiated
    gradients.
    Note that this class does not (yet?) support special
    tensor types such as SparseTensor or RaggedTensor.
    """

    def __init__(
            self,
            coefs: Dict[str, Union[tf.Tensor, tf.IndexedSlices]]
        ) -> None:
        super().__init__(coefs)

    def serialize(
            self,
        ) -> str:
        data = {
            key: self._serialize_tensor(tns)
            for key, tns in self.coefs.items()
        }
        return json.dumps(data)

    @classmethod
    def deserialize(
            cls,
            string: str,
        ) -> 'TensorflowVector':
        data = json.loads(string)
        coef = {
            key: cls._deserialize_tensor(dat)
            for key, dat in data.items()
        }
        return cls(coef)

    @classmethod
    def _serialize_tensor(
            cls,
            tensor: Union[tf.Tensor, tf.IndexedSlices],
        ) -> Tuple[Any, ...]:
        """Convert a Tensor to a JSON-serializable object."""
        if isinstance(tensor, tf.IndexedSlices):
            val = cls._serialize_tensor(tensor.values)
            ind = cls._serialize_tensor(tensor.indices)
            return ("slices", val, ind)
        return serialize_numpy(tensor.numpy())

    @classmethod
    def _deserialize_tensor(
            cls,
            data: Tuple[Any, ...],
        ) -> Union[tf.Tensor, tf.IndexedSlices]:
        """Re-create a Tensor from a JSON-deserialized object."""
        if data[0] == "slices":
            val = cls._deserialize_tensor(data[1])
            ind = cls._deserialize_tensor(data[2])
            return tf.IndexedSlices(val, ind)
        return tf.convert_to_tensor(deserialize_numpy(data))  # type: ignore

    def __eq__(
            self,
            other: Any,
        ) -> bool:
        valid = isinstance(other, TensorflowVector)
        valid &= (self.coefs.keys() == other.coefs.keys())
        return valid and all(
            self._tensor_equal(self.coefs[key], other.coefs[key])
            for key in self.coefs
        )

    @staticmethod
    def _tensor_equal(
            t_a: Union[tf.Tensor, tf.IndexedSlices],
            t_b: Union[tf.Tensor, tf.IndexedSlices],
        ) -> bool:
        if not isinstance(t_a, type(t_b)):
            return False
        if isinstance(t_a, tf.IndexedSlices):
            return (
                TensorflowVector._tensor_equal(t_a.indices, t_b.indices)
                and TensorflowVector._tensor_equal(t_a.values, t_b.values)
            )
        return tf.reduce_all(t_a == t_b).numpy()  # type: ignore

    def _apply_operation(
            self,
            other: Any,
            func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
        ) -> 'TensorflowVector':
        """Apply an operation to combine this vector with another."""
        # Case when operating on two TensorflowVector objects.
        if isinstance(other, (TensorflowVector, NumpyVector)):
            if self.coefs.keys() != other.coefs.keys():
                raise KeyError(
                    f"Cannot {func.__name__} TensorflowVectors "\
                    "with distinct coefficient names."
                )
            return TensorflowVector({
                key: func(self.coefs[key], other.coefs[key])
                for key in self.coefs
            })
        # Case when operating with another object (e.g. a scalar).
        try:
            return TensorflowVector({
                key: func(coef, other)
                for key, coef in self.coefs.items()
            })
        except TypeError as exc:
            raise TypeError(
                f"Cannot {func.__name__} TensorflowVector "\
                f"with object of type {type(other)}."
            ) from exc

    def apply_func(
            self,
            func: Callable[[tf.Tensor], tf.Tensor],
            *args: Any,
            **kwargs: Any
        ) -> 'TensorflowVector':
        """Apply a tensor-altering function to the wrapped coefficients."""
        return TensorflowVector({
            key: func(coef, *args, **kwargs)
            for key, coef in self.coefs.items()
        })

    def __add__(
            self,
            other: Any
        ) -> 'TensorflowVector':
        return self._apply_operation(other, tf.add)

    def __sub__(
            self,
            other: Any
        ) -> 'TensorflowVector':
        return self._apply_operation(other, tf.subtract)

    def __mul__(
            self,
            other: Any
        ) -> 'TensorflowVector':
        return self._apply_operation(other, tf.multiply)

    def __truediv__(
            self,
            other: Any
        ) -> 'TensorflowVector':
        return self._apply_operation(other, tf.divide)

    def __pow__(
            self,
            power: float,
            modulo: Optional[int] = None
        ) -> 'TensorflowVector':
        return self.apply_func(tf.pow, power)

    def sign(
            self
        ) -> 'TensorflowVector':
        return self.apply_func(tf.sign)

    def minimum(
            self,
            other: Any,
        ) -> 'TensorflowVector':
        if isinstance(other, Vector):
            return self._apply_operation(other, tf.minimum)
        return self.apply_func(tf.minimum, other)

    def maximum(
            self,
            other: Any,
        ) -> 'TensorflowVector':
        if isinstance(other, Vector):
            return self._apply_operation(other, tf.maximum)
        return self.apply_func(tf.maximum, other)
