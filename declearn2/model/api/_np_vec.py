# coding: utf-8

"""NumpyVector model coefficients container."""

import json
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from declearn2.model.api._vector import Vector, register_vector_type
from declearn2.utils import deserialize_numpy, serialize_numpy


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

    def __init__(
            self,
            coefs: Dict[str, np.ndarray]
        ) -> None:
        super().__init__(coefs)

    def serialize(
            self,
        ) -> str:
        data = {key: serialize_numpy(arr) for key, arr in self.coefs.items()}
        return json.dumps(data)

    @classmethod
    def deserialize(
            cls,
            string: str,
        ) -> 'NumpyVector':
        data = json.loads(string)
        coef = {key: deserialize_numpy(dat) for key, dat in data.items()}
        return cls(coef)

    def __eq__(
            self,
            other: Any
        ) -> bool:
        valid = isinstance(other, NumpyVector)
        valid &= (self.coefs.keys() == other.coefs.keys())
        return valid and all(
            np.array_equal(self.coefs[k], other.coefs[k]) for k in self.coefs
        )

    def _apply_operation(
            self,
            other: Any,
            func: Callable[[ArrayLike, ArrayLike], ArrayLike]
        ) -> 'NumpyVector':
        """Apply an operation to combine this vector with another."""
        # Case when operating on two NumpyVector objects.
        if isinstance(other, NumpyVector):
            if self.coefs.keys() != other.coefs.keys():
                raise KeyError(
                    f"Cannot {func.__name__} NumpyVectors "\
                    "with distinct coefficient names."
                )
            return NumpyVector({
                key: func(self.coefs[key], other.coefs[key])  # type: ignore
                for key in self.coefs
            })
        # Case when operating with another object (e.g. a scalar).
        try:
            return NumpyVector({
                key: func(coef, other)  # type: ignore
                for key, coef in self.coefs.items()
            })
        except TypeError as exc:
            raise TypeError(
                f"Cannot {func.__name__} NumpyVector "\
                f"with object of type {type(other)}."
            ) from exc

    def apply_ufunc(
            self,
            ufunc: np.ufunc,
            *args: Any,
            **kwargs: Any
        ) -> 'NumpyVector':
        """Apply a numpy ufunc to the wrapped coefficients."""
        if not isinstance(ufunc, np.ufunc):
            raise TypeError(f"Cannot apply non-ufunc object '{ufunc}'")
        return NumpyVector({
            key: ufunc(coef, *args, **kwargs)
            for key, coef in self.coefs.items()
        })

    def __add__(
            self,
            other: Any
        ) -> 'NumpyVector':
        return self._apply_operation(other, np.add)

    def __sub__(
            self,
            other: Any
        ) -> 'NumpyVector':
        return self._apply_operation(other, np.subtract)

    def __mul__(
            self,
            other: Any
        ) -> 'NumpyVector':
        return self._apply_operation(other, np.multiply)

    def __truediv__(
            self,
            other: Any
        ) -> 'NumpyVector':
        return self._apply_operation(other, np.divide)

    def __pow__(
            self,
            power: float,
            modulo: Optional[int] = None
        ) -> 'NumpyVector':
        return self.apply_ufunc(np.power, power)

    def sign(
            self
        ) -> 'NumpyVector':
        return self.apply_ufunc(np.sign)

    def minimum(
            self,
            other: Any,
        ) -> 'Vector':
        if isinstance(other, NumpyVector):
            return self._apply_operation(other, np.minimum)
        return self.apply_ufunc(np.minimum, other)

    def maximum(
            self,
            other: Any,
        ) -> 'Vector':
        if isinstance(other, Vector):
            return self._apply_operation(other, np.maximum)
        return self.apply_ufunc(np.maximum, other)
