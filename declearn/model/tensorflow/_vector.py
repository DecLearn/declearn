# coding: utf-8

"""TensorflowVector gradients container."""

from typing import Any, Dict, Set, Type, Union

import tensorflow as tf  # type: ignore

# false-positive; pylint: disable=no-name-in-module
from tensorflow.python.framework.ops import EagerTensor  # type: ignore

# pylint: enable=no-name-in-module
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn.model.api import NumpyVector, Vector, register_vector_type


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

    _op_add = staticmethod(tf.add)
    _op_sub = staticmethod(tf.subtract)
    _op_mul = staticmethod(tf.multiply)
    _op_div = staticmethod(tf.divide)
    _op_pow = staticmethod(tf.pow)

    @property
    def compatible_vector_types(self) -> Set[Type[Vector]]:
        types = super().compatible_vector_types
        return types.union({NumpyVector, TensorflowVector})

    def __init__(
        self, coefs: Dict[str, Union[tf.Tensor, tf.IndexedSlices]]
    ) -> None:
        super().__init__(coefs)

    def __repr__(
        self,
    ) -> str:
        # fmt: off
        string = f"{type(self).__name__} with {len(self.coefs)} coefs:"
        string += "".join(
            f"\n    {key}: {repr(val.dtype)} slices with shape {val.shape}"
            if isinstance(val, tf.IndexedSlices) else
            f"\n    {key}: {repr(val.dtype)} tensor with shape {val.shape}"
            for key, val in self.coefs.items()
        )
        return string

    def pack(
        self,
    ) -> Dict[str, Any]:
        data = {key: self._pack_tensor(tns) for key, tns in self.coefs.items()}
        return data

    @classmethod
    def unpack(
        cls,
        data: Dict[str, Any],
    ) -> "TensorflowVector":
        coef = {key: cls._unpack_tensor(dat) for key, dat in data.items()}
        return cls(coef)

    @classmethod
    def _pack_tensor(
        cls,
        tensor: Union[tf.Tensor, tf.IndexedSlices],
    ) -> Any:
        """Convert a Tensor to a JSON-serializable object."""
        if isinstance(tensor, tf.IndexedSlices):
            val = cls._pack_tensor(tensor.values)
            ind = cls._pack_tensor(tensor.indices)
            return ["slices", val, ind]
        return tensor.numpy()

    @classmethod
    def _unpack_tensor(
        cls,
        data: Any,
    ) -> Union[tf.Tensor, tf.IndexedSlices]:
        """Re-create a Tensor from a JSON-unpacked object."""
        if isinstance(data, list) and (data[0] == "slices"):
            val = cls._unpack_tensor(data[1])
            ind = cls._unpack_tensor(data[2])
            return tf.IndexedSlices(val, ind)
        try:
            return tf.convert_to_tensor(data)
        except TypeError as exc:
            raise TypeError("Invalid tf.Tensor dump received.") from exc

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        valid = isinstance(other, TensorflowVector)
        valid = valid & (self.coefs.keys() == other.coefs.keys())
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
            return TensorflowVector._tensor_equal(
                t_a.indices, t_b.indices
            ) and TensorflowVector._tensor_equal(t_a.values, t_b.values)
        return tf.reduce_all(t_a == t_b).numpy()  # type: ignore

    def sign(self) -> Self:  # type: ignore
        return self.apply_func(tf.sign)

    def minimum(
        self,
        other: Any,
    ) -> Self:  # type: ignore
        if isinstance(other, Vector):
            return self._apply_operation(other, tf.minimum)
        return self.apply_func(tf.minimum, other)

    def maximum(
        self,
        other: Any,
    ) -> Self:  # type: ignore
        if isinstance(other, Vector):
            return self._apply_operation(other, tf.maximum)
        return self.apply_func(tf.maximum, other)
