# coding: utf-8

"""Vector abstraction API."""

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, Union

from numpy.typing import ArrayLike


__all__ = [
    'Vector',
    'register_vector_type',
]


VECTOR_TYPES = {}  # type: Dict[Type[Any], Type[Vector]]


class Vector(metaclass=ABCMeta):
    """Abstract class defining an API to manipulate (sets of) data arrays.

    A 'Vector' is an abstraction used to wrap a collection of data
    structures (numpy arrays, tensorflow or torch tensors, etc.).
    It enables writing algorithms and operations on such structures,
    agnostic of their actual implementation support.
    """

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

    @abstractmethod
    def __add__(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> 'Vector':
        raise NotImplementedError

    def __radd__(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> 'Vector':
        return self.__add__(other)

    @abstractmethod
    def __sub__(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> 'Vector':
        raise NotImplementedError

    def __rsub__(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> 'Vector':
        return self.__sub__(other)

    @abstractmethod
    def __mul__(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> 'Vector':
        raise NotImplementedError

    def __rmul__(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> 'Vector':
        return self.__mul__(other)

    @abstractmethod
    def __truediv__(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> 'Vector':
        raise NotImplementedError

    def __rtruediv__(
            self,
            other: Union['Vector', float, ArrayLike],
        ) -> Union['Vector', float, ArrayLike]:
        return self.__truediv__(other)

    @abstractmethod
    def __pow__(
            self,
            power: float,
            modulo: Optional[int] = None,
        ) -> 'Vector':
        raise NotImplementedError

    @abstractmethod
    def sign(self) -> 'Vector':
        """Return a Vector storing the sign of each weight."""
        raise NotImplementedError

    @abstractmethod
    def minimum(
            self,
            other: Union['Vector', float, ArrayLike]
        ) -> 'Vector':
        """Compute coef.-wise, element-wise minimum wrt to another Vector."""
        raise NotImplementedError

    @abstractmethod
    def maximum(
            self,
            other: Union['Vector', float, ArrayLike]
        ) -> 'Vector':
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
