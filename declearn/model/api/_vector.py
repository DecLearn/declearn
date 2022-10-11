# coding: utf-8

"""Vector abstraction API."""

import operator
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, Union

from numpy.typing import ArrayLike

# future: `from typing_extensions import Self` and revise Vector return types

from declearn.utils import (
    add_json_support,
    create_types_registry,
    register_type,
)

__all__ = [
    "Vector",
    "register_vector_type",
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

    @staticmethod
    def build(
        coefs: Dict[str, Any],
    ) -> "Vector":
        """Instantiate a Vector, inferring its exact subtype from coefs'.

        'Vector' is an abstract class. Its subclasses, however, are
        expected to be designed for wrapping specific types of data
        structures. Using the `register_vector_type` decorator, the
        implemented Vector subclasses can be made buildable through
        this staticmethod, which relies on input coefficients' type
        analysis to infer the Vector type to instantiate and return.
        """
        # Type-check the inputs and look up the Vector subclass to use.
        if not (isinstance(coefs, dict) and coefs):
            raise TypeError(
                "'Vector.build(coefs)' requires a non-empty 'coefs' dict."
            )
        types = [VECTOR_TYPES.get(type(coef)) for coef in coefs.values()]
        if types[0] is None:
            raise TypeError(
                f"No Vector class was registered for coef. type '{types[0]}'."
            )
        if not all(cls == types[0] for cls in types[1:]):
            raise TypeError(
                "Multiple Vector classes found for input coefficients."
            )
        # Instantiate the Vector subtype and return it.
        return types[0](coefs)

    def __init__(
        self,
        coefs: Dict[str, Any],
    ) -> None:
        """Instantiate the Vector to wrap a collection of data arrays."""
        self.coefs = coefs

    def pack(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this Vector."""
        return self.coefs

    @classmethod
    def unpack(
        cls,
        data: Dict[str, Any],
    ) -> "Vector":
        """Instantiate a Vector from its "packed" dict representation."""
        return cls(data)

    def apply_func(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> "Vector":
        """Apply a given function to the wrapped coefficients."""
        coefs = {
            key: func(coef, *args, **kwargs)
            for key, coef in self.coefs.items()
        }
        return type(self)(coefs)

    def _apply_operation(
        self,
        other: Any,
        func: Callable[[Any, Any], Any],
    ) -> "Vector":
        """Apply an operation to combine this vector with another."""
        # Case when operating on two Vector objects.
        if isinstance(other, type(self)):
            if self.coefs.keys() != other.coefs.keys():
                raise KeyError(
                    f"Cannot {func.__name__} Vectors "
                    "with distinct coefficient names."
                )
            coefs = {
                key: func(self.coefs[key], other.coefs[key])
                for key in self.coefs
            }
            return type(self)(coefs)
        # Case when operating with another object (e.g. a scalar).
        try:
            return type(self)(
                {key: func(coef, other) for key, coef in self.coefs.items()}
            )
        except TypeError as exc:
            raise TypeError(
                f"Cannot {func.__name__} {type(self).__name__} object "
                f"with object of type {type(other)}."
            ) from exc

    def __add__(
        self,
        other: Any,
    ) -> "Vector":
        return self._apply_operation(other, self._op_add)  # type: ignore

    def __radd__(
        self,
        other: Any,
    ) -> "Vector":
        return self.__add__(other)

    def __sub__(
        self,
        other: Any,
    ) -> "Vector":
        return self._apply_operation(other, self._op_sub)  # type: ignore

    def __rsub__(
        self,
        other: Any,
    ) -> "Vector":
        return self.__sub__(-other)

    def __mul__(
        self,
        other: Any,
    ) -> "Vector":
        return self._apply_operation(other, self._op_mul)  # type: ignore

    def __rmul__(
        self,
        other: Any,
    ) -> "Vector":
        return self.__mul__(other)

    def __truediv__(
        self,
        other: Any,
    ) -> "Vector":
        return self._apply_operation(other, self._op_div)  # type: ignore

    def __rtruediv__(
        self,
        other: Any,
    ) -> "Vector":
        return self.__mul__(1 / other)

    def __pow__(
        self,
        other: Any,
    ) -> "Vector":
        return self._apply_operation(other, self._op_pow)  # type: ignore

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def sign(
        self,
    ) -> "Vector":
        """Return a Vector storing the sign of each coefficient."""
        raise NotImplementedError

    @abstractmethod
    def minimum(
        self,
        other: Union["Vector", float, ArrayLike],
    ) -> "Vector":
        """Compute coef.-wise, element-wise minimum wrt to another Vector."""
        raise NotImplementedError

    @abstractmethod
    def maximum(
        self,
        other: Union["Vector", float, ArrayLike],
    ) -> "Vector":
        """Compute coef.-wise, element-wise maximum wrt to another Vector."""
        raise NotImplementedError


create_types_registry("Vector", base=Vector)


def register_vector_type(
    v_type: Type[Any],
    *types: Type[Any],
    name: Optional[str] = None,
) -> Callable[[Type[Vector]], Type[Vector]]:
    """Decorate a Vector subclass to make it buildable with `Vector(...)`.

    Decorating a Vector subclass with this has three effects:
    * Add the class to registered type (in the "Vector" group).
      See `declearn.utils.register_type` for details.
    * Make instances of that class JSON-serializable, embarking
      the wrapped data by using the `pack` and `unpack` methods
      of the class. See `declearn.utils.add_json_support`.
    * Make the subclass buildable through `Vector.build(coefs)`,
      based on the analysis of wrapped coefficients' type.

    Parameters
    ----------
    v_type: type
        Type of wrapped data that is to trigger using `cls`.
    *types: type
        Additional `v_type` alternatives for wrapped data.
    name: str or None, default=None
        Optional name under which to register the type, shared
        by `register_type` and `add_json_support`.
        If None, use `cls.__name__`.

    Returns
    -------
    register: func(cls) -> cls
        A closure that performs the registration operations.
        Hence `register_vector_type` is designed to be used
        as a class decorator.
    """
    v_types = (v_type, *types)
    # Set up a registration function.
    def register(cls: Type[Vector]) -> Type[Vector]:
        nonlocal name, v_types
        if name is None:
            name = cls.__name__
        # Register the Vector type. Note: this type-checks cls.
        register_type(cls, name=name, group="Vector")
        # Add support for JSON (de)serialization, relying on (un)pack.
        add_json_support(cls, cls.pack, cls.unpack, name=name)
        # Make the subclass buildable through `Vector.build(coefs)`.
        for v_type in v_types:
            VECTOR_TYPES[v_type] = cls
        return cls

    # Return the former, enabling decoration syntax for `register_vector_type`.
    return register
