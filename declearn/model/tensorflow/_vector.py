# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""TensorflowVector data arrays container."""

from typing import Any, Callable, Dict, Optional, Set, Type, Union

# fmt: off
import tensorflow as tf  # type: ignore
# false-positive; pylint: disable=no-name-in-module
from tensorflow.python.framework.ops import EagerTensor  # type: ignore
# pylint: enable=no-name-in-module
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn.model.api import Vector, register_vector_type
from declearn.model.sklearn import NumpyVector

# fmt: on


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

    Use `vector.coefs` to access the stored coefficients.
    """

    @property
    def _op_add(self) -> Callable[[Any, Any], Any]:
        return tf.add

    @property
    def _op_sub(self) -> Callable[[Any, Any], Any]:
        return tf.subtract

    @property
    def _op_mul(self) -> Callable[[Any, Any], Any]:
        return tf.multiply

    @property
    def _op_div(self) -> Callable[[Any, Any], Any]:
        return tf.divide

    @property
    def _op_pow(self) -> Callable[[Any, Any], Any]:
        return tf.pow

    @property
    def compatible_vector_types(self) -> Set[Type[Vector]]:
        types = super().compatible_vector_types
        return types.union({NumpyVector, TensorflowVector})

    def __init__(
        self, coefs: Dict[str, Union[tf.Tensor, tf.IndexedSlices]]
    ) -> None:
        super().__init__(coefs)

    def dtypes(
        self,
    ) -> Dict[str, str]:
        return {key: coef.dtype.name for key, coef in self.coefs.items()}

    def pack(
        self,
    ) -> Dict[str, Any]:
        data = {key: self._pack_tensor(tns) for key, tns in self.coefs.items()}
        return data

    @classmethod
    def unpack(
        cls,
        data: Dict[str, Any],
    ) -> Self:
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
        if valid:
            valid = self.coefs.keys() == other.coefs.keys()
        if valid:
            valid = all(
                self._tensor_equal(self.coefs[key], other.coefs[key])
                for key in self.coefs
            )
        return valid

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
        return tf.reduce_all(t_a == t_b).numpy()

    def sign(self) -> Self:
        return self.apply_func(tf.sign)

    def minimum(
        self,
        other: Any,
    ) -> Self:
        if isinstance(other, Vector):
            return self._apply_operation(other, tf.minimum)
        return self.apply_func(tf.minimum, other)

    def maximum(
        self,
        other: Any,
    ) -> Self:
        if isinstance(other, Vector):
            return self._apply_operation(other, tf.maximum)
        return self.apply_func(tf.maximum, other)

    def sum(
        self,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> Self:
        coefs = {
            key: tf.reduce_sum(val, axis=axis, keepdims=keepdims)
            for key, val in self.coefs.items()
        }
        return self.__class__(coefs)
