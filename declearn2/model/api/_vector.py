# coding: utf-8

"""Vector abstraction API."""

import operator
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Type, Union

from numpy.typing import ArrayLike
from typing_extensions import Self  # future: import from typing (Py>=3.11)


__all__ = [
    'Vector',
    'register_vector_type',
]


VECTOR_TYPES = {}  # type: Dict[Type[Any], Type[Vector]]


class Vector(metaclass=ABCMeta):
    """Abstract class defining an API to manipulate (sets of) data arrays.

    A Vector is an abstraction used to wrap a collection of data
    structures (numpy arrays, tensorflow or torch tensors, etc.).
    It enables writing algorithms and operations on such structures,
    agnostic of their actual implementation support.
    """

    _op_add = operator.add
    _op_sub = operator.sub
    _op_mul = operator.mul
    _op_div = operator.truediv
    _op_pow = operator.pow

    def __new__(
            cls,
            coefs: Dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> 'Vector':
        # Case: generic Vector builder on a supported tensor type.
        if (cls is Vector) and coefs:
            arr = list(coefs.values())[0]
            sub_cls = VECTOR_TYPES.get(type(arr))
            if sub_cls is not None:
                return sub_cls(coefs, *args, **kwargs)
        # Case: specific Vector builder (or invalid call).
        return object.__new__(cls)

    def __init__(
            self,
            coefs: Dict[str, Any],
        ) -> None:
        """Instantiate the Vector to wrap a collection of data arrays.

        Note: depending on the type of coefficients, calling
              `Vector(coefs)` directly will return a suitable
              Vector subclass's instance.
        """
        self.coefs = coefs

    @abstractmethod
    def serialize(
            self,
        ) -> str:
        """Serialize this Vector to a string."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def deserialize(
            cls,
            string: str,
        ) -> 'Vector':
        """Deserialize a Vector from a string."""
        raise NotImplementedError

    def apply_func(
            self,
            func: Callable[..., Any],
            *args: Any,
            **kwargs: Any
        ) -> Self:  # type: ignore
        """Apply a given function to the wrapped coefficients."""
        return type(self)({
            key: func(coef, *args, **kwargs)
            for key, coef in self.coefs.items()
        })

    def _apply_operation(
            self,
            other: Any,
            func: Callable[[Any, Any], Any],
        ) -> Self:  # type: ignore
        """Apply an operation to combine this vector with another."""
        # Case when operating on two Vector objects.
        if isinstance(other, type(self)):
            if self.coefs.keys() != other.coefs.keys():
                raise KeyError(
                    f"Cannot {func.__name__} Vectors "\
                    "with distinct coefficient names."
                )
            return type(self)({
                key: func(self.coefs[key], other.coefs[key])
                for key in self.coefs
            })
        # Case when operating with another object (e.g. a scalar).
        try:
            return type(self)({
                key: func(coef, other)
                for key, coef in self.coefs.items()
            })
        except TypeError as exc:
            raise TypeError(
                f"Cannot {func.__name__} {type(self).__name__} object "\
                f"with object of type {type(other)}."
            ) from exc

    def __add__(
            self,
            other: Any,
        ) -> Self:  # type: ignore
        return self._apply_operation(other, self._op_add)  # type: ignore

    def __radd__(
            self,
            other: Any,
        ) -> Self:  # type: ignore
        return self.__add__(other)

    def __sub__(
            self,
            other: Any,
        ) -> Self:  # type: ignore
        return self._apply_operation(other, self._op_sub)  # type: ignore

    def __rsub__(
            self,
            other: Any,
        ) -> Self:  # type: ignore
        return self.__sub__(- other)

    def __mul__(
            self,
            other: Any,
        ) -> Self:  # type: ignore
        return self._apply_operation(other, self._op_mul)  # type: ignore

    def __rmul__(
            self,
            other: Any,
        ) -> Self:  # type: ignore
        return self.__mul__(other)

    def __truediv__(
            self,
            other: Any,
        ) -> Self:  # type: ignore
        return self._apply_operation(other, self._op_div)  # type: ignore

    def __rtruediv__(
            self,
            other: Any,
        ) -> Self:  # type: ignore
        return self.__mul__(1 / other)

    def __pow__(
            self,
            other: Any,
        ) -> Self:  # type: ignore
        return self._apply_operation(other, self._op_pow)  # type: ignore

    @abstractmethod
    def __eq__(
            self,
            other: Any
        ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def sign(
            self,
        ) -> Self:  # type: ignore
        """Return a Vector storing the sign of each coefficient."""
        raise NotImplementedError

    @abstractmethod
    def minimum(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> Self:  # type: ignore
        """Compute coef.-wise, element-wise minimum wrt to another Vector."""
        raise NotImplementedError

    @abstractmethod
    def maximum(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> Self:  # type: ignore
        """Compute coef.-wise, element-wise maximum wrt to another Vector."""
        raise NotImplementedError


def register_vector_type(
        v_type: Type[Any],
        *types: Type[Any],
    ) -> Callable[[Type[Vector]], Type[Vector]]:
    """Decorate a Vector subclass to make it buildable with `Vector(...)`.

    `Vector` is an abstract class that cannot be instantiated as such.
    However, registering subtypes with this decorator enables using
    the generic syntax `Vector(coefs)` as an alias for `cls(coefs)`
    provided the wrapped coefficients are of a proper (registered)
    type.

    v_type: type
        Type of wrapped data that is to trigger using `cls`.
    *types: type
        Additional `v_type` alternatives for wrapped data.
    """
    v_types = (v_type, *types)
    # Set up a registration function.
    def register(cls: Type[Vector]) -> Type[Vector]:
        nonlocal v_types
        if not issubclass(cls, Vector):
            raise TypeError(
                f"Cannot register non-Vector class '{cls}' as a Vector type."
            )
        for v_type in v_types:
            VECTOR_TYPES[v_type] = cls
        return cls
    # Return the former, enabling decoration syntax for `register_vector_type`.
    return register
