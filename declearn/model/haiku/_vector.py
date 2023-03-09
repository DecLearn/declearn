# coding: utf-8

"""JaxNumpyVector data arrays container."""

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config as jaxconfig
from jaxtyping import Array
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn.model.api._vector import Vector, register_vector_type
from declearn.model.haiku.utils import select_device
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

    def __init__(self, coefs: Dict[str, Array]) -> None:
        super().__init__(coefs)

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
