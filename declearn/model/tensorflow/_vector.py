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
# fmt: on

from declearn.model.api import Vector, register_vector_type
from declearn.model.sklearn import NumpyVector
from declearn.model.tensorflow.utils import (
    preserve_tensor_device,
    select_device,
)
from declearn.utils import get_device_policy


@register_vector_type(tf.Tensor, EagerTensor, tf.IndexedSlices)
class TensorflowVector(Vector):
    """Vector subclass to store tensorflow tensors.

    This Vector is designed to store a collection of named TensorFlow
    tensors, enabling computations that are either applied to each and
    every coefficient, or imply two sets of aligned coefficients (i.e.
    two TensorflowVector with similar specifications).

    Note that support for IndexedSlices is implemented, as these are a
    common type for auto-differentiated gradients.

    Note that this class does not (yet?) support special tensor types
    such as SparseTensor or RaggedTensor.

    Use `vector.coefs` to access the stored coefficients.

    Notes
    -----
    - A `TensorflowVector` can be operated with either a:
      - scalar value
      - `NumpyVector` that has similar specifications
      - `TensorflowVector` that has similar specifications
      => resulting in a `TensorflowVector` in each of these cases.
    - The wrapped tensors may be placed on any device (CPU, GPU...)
      and may not be all on the same device.
    - The device-placement of the initial `TensorflowVector`'s data
      is preserved by operations, including with `NumpyVector`.
    - When combining two `TensorflowVector`, the device-placement
      of the left-most one is used; in that case, one ends up with
      `gpu + cpu = gpu` while `cpu + gpu = cpu`. In both cases, a
      warning will be emitted to prevent silent un-optimized copies.
    - When deserializing a `TensorflowVector` (either by directly using
      `TensorflowVector.unpack` or loading one from a JSON dump), loaded
      tensors are placed based on the global device-placement policy
      (accessed via `declearn.utils.get_device_policy`). Thus it may
      have a different device-placement schema than at dump time but
      should be coherent with that of `TensorflowModel` computations.
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

    def apply_func(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        func = preserve_tensor_device(func)
        return super().apply_func(func, *args, **kwargs)

    def _apply_operation(
        self,
        other: Any,
        func: Callable[[Any, Any], Any],
    ) -> Self:
        func = preserve_tensor_device(func)
        return super()._apply_operation(other, func)

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
        policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        with tf.device(device):
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
            # fmt: off
            return (
                TensorflowVector._tensor_equal(t_a.indices, t_b.indices)
                and TensorflowVector._tensor_equal(t_a.values, t_b.values)
            )
        with tf.device(t_a.device):
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
        return self.apply_func(tf.reduce_sum, axis=axis, keepdims=keepdims)

    def __pow__(
        self,
        other: Any,
    ) -> Self:
        # For square and square root, use dedicated functions rather
        # than tf.pow as results tend to differ for small values.
        if isinstance(other, (int, float)):
            if other == 2:
                return self.apply_func(tf.square)
            if other == 0.5:
                return self.apply_func(tf.sqrt)
        return super().__pow__(other)
