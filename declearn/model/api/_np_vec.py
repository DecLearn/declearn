# coding: utf-8

"""NumpyVector model coefficients container."""

from typing import Any, Callable, Dict, Union

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn.model.api._vector import Vector, register_vector_type


__all__ = [
    "NumpyVector",
]


@register_vector_type(np.ndarray)
class NumpyVector(Vector):
    """Vector subclass to store numpy.ndarray coefficients.

    This Vector is designed to store a collection of named
    numpy arrays or scalars, enabling computations that are
    either applied to each and every coefficient, or imply
    two sets of aligned coefficients (i.e. two NumpyVector
    instances with similar coefficients specifications).

    Use `vector.coefs` to access the stored coefficients.
    """

    @property
    def _op_add(self) -> Callable[[Any, Any], Any]:
        return np.add

    @property
    def _op_sub(self) -> Callable[[Any, Any], Any]:
        return np.subtract

    @property
    def _op_mul(self) -> Callable[[Any, Any], Any]:
        return np.multiply

    @property
    def _op_div(self) -> Callable[[Any, Any], Any]:
        return np.divide

    @property
    def _op_pow(self) -> Callable[[Any, Any], Any]:
        return np.power

    def __init__(self, coefs: Dict[str, np.ndarray]) -> None:
        super().__init__(coefs)

    def __eq__(self, other: Any) -> bool:
        valid = isinstance(other, NumpyVector)
        valid = valid and (self.coefs.keys() == other.coefs.keys())
        return valid and all(
            np.array_equal(self.coefs[k], other.coefs[k]) for k in self.coefs
        )

    def sign(
        self,
    ) -> Self:  # type: ignore
        return self.apply_func(np.sign)

    def minimum(
        self,
        other: Union["Vector", float, ArrayLike],
    ) -> Self:  # type: ignore
        if isinstance(other, NumpyVector):
            return self._apply_operation(other, np.minimum)
        return self.apply_func(np.minimum, other)

    def maximum(
        self,
        other: Union["Vector", float, ArrayLike],
    ) -> Self:  # type: ignore
        if isinstance(other, Vector):
            return self._apply_operation(other, np.maximum)
        return self.apply_func(np.maximum, other)
