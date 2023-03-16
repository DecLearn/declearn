# coding: utf-8

"""JaxNumpyVector data arrays container."""

from typing import Any, Callable, Dict, Optional, Set, Type

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.config import config as jaxconfig
from jax.typing import ArrayLike
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn.model.api import Vector, register_vector_type
from declearn.model.haiku.utils import select_device
from declearn.model.sklearn import NumpyVector
from declearn.utils import get_device_policy

__all__ = [
    "JaxNumpyVector",
]

# Overriding float32 default in jax
jaxconfig.update("jax_enable_x64", True)


@register_vector_type(Array)
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
      => resulting in a `JaxNumpyVector` in each of these cases.
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
    def _op_add(self) -> Callable[[Any, Any], Any]:
        return jnp.add

    @property
    def _op_sub(self) -> Callable[[Any, Any], Any]:
        return jnp.subtract

    @property
    def _op_mul(self) -> Callable[[Any, Any], Any]:
        return jnp.multiply

    @property
    def _op_div(self) -> Callable[[Any, Any], Any]:
        return jnp.divide

    @property
    def _op_pow(self) -> Callable[[Any, Any], Any]:
        return jnp.power

    @property
    def compatible_vector_types(self) -> Set[Type[Vector]]:
        types = super().compatible_vector_types
        return types.union({NumpyVector, JaxNumpyVector})

    def __init__(self, coefs: Dict[str, ArrayLike]) -> None:
        super().__init__(coefs)

    def _apply_operation(
        self,
        other: Any,
        func: Callable[[Any, Any], Any],
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
    ) -> Self:  # type: ignore
        return self.apply_func(jnp.sign)

    def minimum(
        self,
        other: Any,
    ) -> Self:  # type: ignore
        if isinstance(other, JaxNumpyVector):
            return self._apply_operation(other, jnp.minimum)
        return self.apply_func(jnp.minimum, other)

    def maximum(
        self,
        other: Any,
    ) -> Self:  # type: ignore
        if isinstance(other, Vector):
            return self._apply_operation(other, jnp.maximum)
        return self.apply_func(jnp.maximum, other)

    def sum(
        self,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> Self:
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
