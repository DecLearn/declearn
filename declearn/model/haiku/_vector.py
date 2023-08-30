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

"""JaxNumpyVector data arrays container."""

import warnings
from typing import Any, Callable, Dict, Optional, Set, Type, Union

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn.model.api import Vector, register_vector_type
from declearn.model.haiku.utils import select_device
from declearn.model.sklearn import NumpyVector
from declearn.utils import get_device_policy

__all__ = [
    "JaxNumpyVector",
]


jax.config.update("jax_enable_x64", True)  # enable float64 support


@register_vector_type(
    jax.Array,
    jaxlib.xla_extension.ArrayImpl,  # pylint: disable=c-extension-no-member
)
class JaxNumpyVector(Vector):
    """Vector subclass to store jax.numpy.ndarray coefficients.

    This Vector is designed to store a collection of named
    jax numpy arrays or scalars, enabling computations that are
    either applied to each and every coefficient, or imply
    two sets of aligned coefficients (i.e. two JaxNumpyVector
    instances with similar coefficients specifications).

    Use `vector.coefs` to access the stored coefficients.

    Notes
    -----
    - A `JaxnumpyVector` can be operated with either a:
        - scalar value
        - `NumpyVector` that has similar specifications
        - `JaxNumpyVector` that has similar specifications
        - => resulting in a `JaxNumpyVector` in each of these cases.
    - The wrapped arrays may be placed on any device (CPU, GPU...)
      and may not be all on the same device.
    - The device-placement of the initial `JaxNumpyVector`'s data
      is preserved by operations, including with `NumpyVector`.
    - When combining two `JaxNumpyVector`, the device-placement
      of the left-most one is used; in that case, one ends up with
      `gpu + cpu = gpu` while `cpu + gpu = cpu`. In both cases, a
      warning will be emitted to prevent silent un-optimized copies.
    - When deserializing a `JaxNumpyVector` (either by directly using
      `JaxNumpyVector.unpack` or loading one from a JSON dump), loaded
      arrays are placed based on the global device-placement policy
      (accessed via `declearn.utils.get_device_policy`). Thus it may
      have a different device-placement schema than at dump time but
      should be coherent with that of `HaikuModel` computations.
    """

    @property
    def _op_add(self) -> Callable[[Any, Any], jax.Array]:
        return jnp.add

    @property
    def _op_sub(self) -> Callable[[Any, Any], jax.Array]:
        return jnp.subtract

    @property
    def _op_mul(self) -> Callable[[Any, Any], jax.Array]:
        return jnp.multiply

    @property
    def _op_div(self) -> Callable[[Any, Any], jax.Array]:
        return jnp.divide

    @property
    def _op_pow(self) -> Callable[[Any, Any], jax.Array]:
        return jnp.power

    @property
    def compatible_vector_types(self) -> Set[Type[Vector]]:
        types = super().compatible_vector_types
        return types.union({NumpyVector, JaxNumpyVector})

    def __init__(self, coefs: Dict[str, jax.Array]) -> None:
        super().__init__(coefs)

    def _apply_operation(
        self,
        other: Any,
        func: Callable[[jax.Array, Any], jax.Array],
    ) -> Self:
        # Ensure 'other' JaxNumpyVector shares this vector's device placement.
        if isinstance(other, JaxNumpyVector):
            coefs = {
                key: jax.device_put(val, self.coefs[key].device())
                for key, val in other.coefs.items()
            }
            other = JaxNumpyVector(coefs)
        return super()._apply_operation(other, func)

    def __eq__(self, other: Any) -> bool:
        valid = isinstance(other, JaxNumpyVector)
        valid = valid and (self.coefs.keys() == other.coefs.keys())
        return valid and all(
            jnp.array_equal(self.coefs[k], other.coefs[k]) for k in self.coefs
        )

    def sign(
        self,
    ) -> Self:
        return self.apply_func(jnp.sign)

    def minimum(
        self,
        other: Union[Self, float],
    ) -> Self:
        if isinstance(other, JaxNumpyVector):
            return self._apply_operation(other, jnp.minimum)
        return self.apply_func(jnp.minimum, other)

    def maximum(
        self,
        other: Union[Self, float],
    ) -> Self:
        if isinstance(other, Vector):
            return self._apply_operation(other, jnp.maximum)
        return self.apply_func(jnp.maximum, other)

    def sum(
        self,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> Self:
        if isinstance(axis, int) or keepdims:
            warnings.warn(  # pragma: no cover
                "The 'axis' and 'keepdims' arguments of 'JaxNumpyVector.sum' "
                "have been deprecated as of declearn v2.3, and will be "
                "removed in version 2.5 and/or 3.0.",
                DeprecationWarning,
            )
        coefs = {
            key: jnp.array(jnp.sum(val, axis=axis, keepdims=keepdims))
            for key, val in self.coefs.items()
        }
        return self.__class__(coefs)

    def pack(
        self,
    ) -> Dict[str, Any]:
        return {key: np.asarray(arr) for key, arr in self.coefs.items()}

    @classmethod
    def unpack(
        cls,
        data: Dict[str, Any],
    ) -> Self:
        policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        coefs = {k: jax.device_put(arr, device) for k, arr in data.items()}
        return cls(coefs)
