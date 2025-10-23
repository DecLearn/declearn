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

"""TensorflowVector data arrays container."""

import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Self,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# fmt: off
import numpy as np
import tensorflow as tf  # type: ignore

# false-positive; pylint: disable=no-name-in-module
from tensorflow.python.framework.ops import EagerTensor  # type: ignore

from declearn.model._utils import flatten_numpy_arrays, unflatten_numpy_arrays

# fmt: on
from declearn.model.api import Vector, VectorSpec, register_vector_type
from declearn.model.sklearn import NumpyVector
from declearn.model.tensorflow.utils import (
    add_indexed_slices_support,
    preserve_tensor_device,
    select_device,
)
from declearn.utils import get_device_policy

# pylint: enable=no-name-in-module


__all__ = [
    "TensorflowVector",
]


TensorT = TypeVar("TensorT", tf.Tensor, tf.IndexedSlices)


def enhance_tf_op(
    tf_op: Callable[[tf.Tensor, Any], tf.Tensor],
    inplc: bool = False,
) -> Callable[[TensorT, Any], TensorT]:
    """Wrap up a tensorflow operation to preserve IndexedSlices and device."""
    func = add_indexed_slices_support(preserve_tensor_device(tf_op), inplc)
    func._pre_wrapped = True  # type: ignore
    return func


# Wrap up base tensorflow operations to add support for IndexedSlices
# inputs and preserve tensor's device-placement
tf_op_add = enhance_tf_op(tf.add)
tf_op_sub = enhance_tf_op(tf.subtract)
tf_op_mul = enhance_tf_op(tf.multiply)
tf_op_div = enhance_tf_op(tf.truediv)
tf_op_pow = enhance_tf_op(tf.pow)
tf_op_min = enhance_tf_op(tf.minimum)
tf_op_max = enhance_tf_op(tf.maximum)
tf_op_sign = enhance_tf_op(tf.sign, inplc=True)
tf_op_sqre = enhance_tf_op(tf.square, inplc=True)
tf_op_sqrt = enhance_tf_op(tf.sqrt, inplc=True)


@register_vector_type(tf.Tensor, EagerTensor, tf.IndexedSlices)
class TensorflowVector(Vector):  # noqa : PLW1641
    """Vector subclass to store tensorflow tensors.

    This Vector is designed to store a collection of named TensorFlow
    tensors, enabling computations that are either applied to each and
    every coefficient, or imply two sets of aligned coefficients (i.e.
    two TensorflowVector with similar specifications).

    Note that support for IndexedSlices is implemented, as these are a
    common type for auto-differentiated gradients. When using built-in
    operators and methods, these structures will be preserved, unless
    densification is required (e.g. when summing with a dense tensor).
    When using `TensorflowVector.apply_func` directly, support for the
    IndexedSlices' preservation should be added manually, typically by
    using `declearn.model.tensorflow.utils.add_indexed_slices_support`.

    Note that this class does not currently support special tensor types
    such as SparseTensor or RaggedTensor.

    Use `vector.coefs` to access the stored coefficients.

    Notes
    -----
    - A `TensorflowVector` can be operated with either a:
        - scalar value
        - `NumpyVector` that has similar specifications
        - `TensorflowVector` that has similar specifications
        - => resulting in a `TensorflowVector` in each of these cases.
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
        return tf_op_add

    @property
    def _op_sub(self) -> Callable[[Any, Any], Any]:
        return tf_op_sub

    @property
    def _op_mul(self) -> Callable[[Any, Any], Any]:
        return tf_op_mul

    @property
    def _op_div(self) -> Callable[[Any, Any], Any]:
        return tf_op_div

    @property
    def _op_pow(self) -> Callable[[Any, Any], Any]:
        return tf_op_pow

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
        if not getattr(func, "_pre_wrapped", False):
            func = preserve_tensor_device(func)  # pragma: no cover
        return super().apply_func(func, *args, **kwargs)

    def _apply_operation(
        self,
        other: Any,
        func: Callable[[Any, Any], Any],
    ) -> Self:
        if not getattr(func, "_pre_wrapped", False):
            func = preserve_tensor_device(func)  # pragma: no cover
        return super()._apply_operation(other, func)

    def shapes(
        self,
    ) -> Dict[str, Tuple[int, ...]]:
        return {
            key: tuple(coef.shape.as_list())
            for key, coef in self.coefs.items()
        }

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
            shp = cls._pack_tensor(tensor.dense_shape)  # type: ignore
            return ["slices", val, ind, shp]
        return np.array(tensor.numpy())

    @classmethod
    def _unpack_tensor(
        cls,
        data: Any,
    ) -> Union[tf.Tensor, tf.IndexedSlices]:
        """Re-create a Tensor from a JSON-unpacked object."""
        if isinstance(data, list) and (data[0] == "slices"):
            val = cls._unpack_tensor(data[1])
            ind = cls._unpack_tensor(data[2])
            shp = cls._unpack_tensor(data[3])
            return tf.IndexedSlices(val, ind, shp)  # type: ignore
        try:
            return tf.convert_to_tensor(data)
        except TypeError as exc:  # pragma: no cover
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
        return self.apply_func(tf_op_sign)

    def minimum(
        self,
        other: Union[Self, float],
    ) -> Self:
        if isinstance(other, Vector):
            return self._apply_operation(other, tf_op_min)
        return self.apply_func(tf_op_min, other)

    def maximum(
        self,
        other: Union[Self, float],
    ) -> Self:
        if isinstance(other, Vector):
            return self._apply_operation(other, tf_op_max)
        return self.apply_func(tf_op_max, other)

    def sum(
        self,
    ) -> Self:
        return self.apply_func(tf.reduce_sum)

    def __pow__(
        self,
        other: Any,
    ) -> Self:
        # For square and square root, use dedicated functions rather
        # than tf.pow as results tend to differ for small values.
        if isinstance(other, (int, float)):
            if other == 2:
                return self.apply_func(tf_op_sqre)
            if other == 0.5:
                return self.apply_func(tf_op_sqrt)
        return super().__pow__(other)

    def get_vector_specs(
        self,
    ) -> VectorSpec:
        # Add IndexedSlices information to the base specs.
        specs = super().get_vector_specs()
        slices = [
            name
            for name in specs.names
            if isinstance(self.coefs[name], tf.IndexedSlices)
        ]
        if slices:
            specs.kwargs["slices"] = slices
        return specs

    def flatten(
        self,
    ) -> Tuple[List[float], VectorSpec]:
        v_spec = self.get_vector_specs()
        arrays: List[np.ndarray] = []
        for name in v_spec.names:
            if isinstance(self.coefs[name], tf.IndexedSlices):
                warnings.warn(
                    "Flattening a 'tf.IndexedSlices' structure into an "
                    "exhaustive list of floats. This may result in a high "
                    "memory cost.",
                    UserWarning,
                    stacklevel=2,
                )
                arrays.append(
                    np.array(tf.convert_to_tensor(self.coefs[name]).numpy())
                )
            else:
                arrays.append(np.array(self.coefs[name].numpy()))
        values = flatten_numpy_arrays(arrays)
        return values, v_spec

    @classmethod
    def unflatten(
        cls,
        values: List[float],
        v_spec: VectorSpec,
    ) -> Self:
        # Convert values to numpy arrays.
        shapes = [v_spec.shapes[name] for name in v_spec.names]
        dtypes = [v_spec.dtypes[name] for name in v_spec.names]
        arrays = unflatten_numpy_arrays(values, shapes, dtypes)
        # Gather information about IndexedSlices and prepare a list of tensors.
        slices = v_spec.kwargs.get("slices", [])
        tf_dat = {}  # Dict[str, Union[tf.Tensor, tf.IndexedSlices]]
        # Convert numpy arrays back to tensorflow structures.
        # Operate under the current device-placement policy.
        policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        with tf.device(device):
            for name, array in zip(v_spec.names, arrays, strict=False):
                # Recover IndexedSlices structures from dense arrays.
                if name in slices:
                    non_zero = np.any(
                        array != 0.0, axis=tuple(range(1, array.ndim))
                    )
                    ind = np.where(non_zero)[0].astype("int32")
                    tf_dat[name] = tf.IndexedSlices(
                        values=tf.convert_to_tensor(array[non_zero]),
                        indices=tf.convert_to_tensor(ind),
                        dense_shape=tf.convert_to_tensor(array.shape),
                    )
                # Otherwise, merely convert arrays to tensors.
                else:
                    tf_dat[name] = tf.convert_to_tensor(array)  # type: ignore
        return cls(tf_dat)  # type: ignore
