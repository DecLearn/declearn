# coding: utf-8

"""NumpyVector model coefficients container."""

from typing import Any, Dict, Union

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn.model.api._vector import Vector, register_vector_type


__all__ = [
    'NumpyVector',
]

@register_vector_type(np.ndarray)
class NumpyVector(Vector):
    """Vector subclass to store numpy.ndarray coefficients.

    This Vector is designed to store a collection of named
    numpy arrays or scalars, enabling computations that are
    either applied to each and every coefficient, or imply
    two sets of aligned coefficients (i.e. two NumpyVector
    instances with similar coefficients specifications).
    """

    _op_add = np.add
    _op_sub = np.subtract
    _op_mul = np.multiply
    _op_truediv = np.divide
    _op_pow = np.power

    def __init__(
            self,
            coefs: Dict[str, np.ndarray]
        ) -> None:
        super().__init__(coefs)

    def __repr__(
            self,
        ) -> str:
        string = f"{type(self).__name__} with {len(self.coefs)} coefs:"
        string += "".join(
            f"\n    {key}: {val.dtype} array with shape {val.shape}"
            for key, val in self.coefs.items()
        )
        return string

    def __eq__(
            self,
            other: Any
        ) -> bool:
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
            other: Union['Vector', float, ArrayLike],
        ) -> Self:  # type: ignore
        if isinstance(other, NumpyVector):
            return self._apply_operation(other, np.minimum)
        return self.apply_func(np.minimum, other)

    def maximum(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> Self:  # type: ignore
        if isinstance(other, Vector):
            return self._apply_operation(other, np.maximum)
        return self.apply_func(np.maximum, other)
