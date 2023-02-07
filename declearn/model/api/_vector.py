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

"""Vector abstraction API."""

import operator
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union

from numpy.typing import ArrayLike
from typing_extensions import Self  # future: import from typing (Py>=3.11)

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


@create_types_registry
class Vector(metaclass=ABCMeta):
    """Abstract class defining an API to manipulate (sets of) data arrays.

    A Vector is an abstraction used to wrap a collection of data
    structures (numpy arrays, tensorflow or torch tensors, etc.).
    It enables writing algorithms and operations on such structures,
    agnostic of their actual implementation support.

    Use `vector.coefs` to access the stored coefficients.

    Any concrete Vector subclass should:
    - add type checks to `__init__` to control wrapped coefficients' type
    - opt. override `_op_...` properties to define compatible operators
    - implement the abstract operators (`sign`, `maximum`, `minimum`...)
    - opt. override `pack` and `unpack` to enable their serialization
    - opt. extend `compatible_vector_types` to specify their compatibility
      with other Vector subclasses
    - opt. override the `dtypes` and `shapes` methods
    """

    @property
    def _op_add(self) -> Callable[[Any, Any], Any]:
        """Framework-compatible addition operator."""
        return operator.add

    @property
    def _op_sub(self) -> Callable[[Any, Any], Any]:
        """Framework-compatible substraction operator."""
        return operator.sub

    @property
    def _op_mul(self) -> Callable[[Any, Any], Any]:
        """Framework-compatible multiplication operator."""
        return operator.mul

    @property
    def _op_div(self) -> Callable[[Any, Any], Any]:
        """Framework-compatible true division operator."""
        return operator.truediv

    @property
    def _op_pow(self) -> Callable[[Any, Any], Any]:
        """Framework-compatible power operator."""
        return operator.pow

    @property
    def compatible_vector_types(self) -> Set[Type["Vector"]]:
        """Compatible Vector types, that may be combined into this.

        Note that VectorTypeA may be compatible with VectorTypeB
        while the opposite is False. It means that, for example,
            (VectorTypeB + VectorTypeA) -> VectorTypeB
        while
            (VectorTypeA + VectorTypeB) -> TypeError

        This is for example the case is VectorTypeB stores numpy
        arrays while VectorTypeA stores tensorflow tensors since
        tf.add(tensor, array) returns a tensor, not an array.
        """
        return {type(self)}

    def __init__(
        self,
        coefs: Dict[str, Any],
    ) -> None:
        """Instantiate the Vector to wrap a collection of data arrays.

        Parameters
        ----------
        coefs: dict[str, any]
            Dict grouping a named collection of data arrays.
            The supported types of that dict's values depends
            on the concrete `Vector` subclass being used.
        """
        self.coefs = coefs

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

        Parameters
        ----------
        coefs: dict[str, any]
            Dict grouping a named collection of data arrays, that
            all belong to the same framework.

        Returns
        -------
        vector: Vector
            Vector instance, the concrete class of which depends
            on that of the values of the `coefs` dict.
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

    def __repr__(self) -> str:
        string = f"{type(self).__name__} with {len(self.coefs)} coefs:"
        dtypes = self.dtypes()
        shapes = self.shapes()
        otypes = {
            key: f"{type(val).__module__}.{type(val).__name__}"
            for key, val in self.coefs.items()
        }
        string += "".join(
            f"\n    {k}: {dtypes[k]} {otypes[k]} with shape {shapes[k]}"
            for k in self.coefs
        )
        return string

    def shapes(
        self,
    ) -> Dict[str, Tuple[int, ...]]:
        """Return a dict storing the shape of each coefficient.

        Returns
        -------
        shapes: dict[str, tuple(int, ...)]
            Dict containing the shape of each of the wrapped data array,
            indexed by the coefficient's name.
        """
        try:
            return {key: coef.shape for key, coef in self.coefs.items()}
        except AttributeError as exc:
            raise NotImplementedError(
                "Wrapped coefficients appear not to implement `.shape`.\n"
                f"`{type(self).__name__}.shapes` probably needs overriding."
            ) from exc

    def dtypes(
        self,
    ) -> Dict[str, str]:
        """Return a dict storing the dtype of each coefficient.

        Returns
        -------
        dtypes: dict[str, tuple(int, ...)]
            Dict containing the dtype of each of the wrapped data array,
            indexed by the coefficient's name. The dtypes are parsed as
            a string, the values of which may vary depending on the
            concrete framework of the Vector.
        """
        try:
            return {key: str(coef.dtype) for key, coef in self.coefs.items()}
        except AttributeError as exc:
            raise NotImplementedError(
                "Wrapped coefficients appear not to implement `.dtype`.\n"
                f"`{type(self).__name__}.dtypes` probably needs overriding."
            ) from exc

    def pack(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this Vector.

        This method must return a dict that can be serialized to and from
        JSON using the JSON-extending declearn hooks (see `json_pack` and
        `json_unpack` functions from the `declearn.utils` module).

        The counterpart `unpack` method may be used to re-create a Vector
        from its "packed" dict representation.

        Returns
        -------
        packed: dict[str, any]
            Dict with str keys, that may be serialized to and from JSON
            using the `declearn.utils.json_pack` and `json_unpack` util
            functions.
        """
        return self.coefs

    @classmethod
    def unpack(
        cls,
        data: Dict[str, Any],
    ) -> Self:
        """Instantiate a Vector from its "packed" dict representation.

        This method is the counterpart to the `pack` one.

        Parameters
        ----------
        data: dict[str, any]
            Dict produced by the `pack` method of an instance of this class.

        Returns
        -------
        vector: Self
            Instance of this Vector subclass, (re-)created from the inputs.
        """
        return cls(data)

    def apply_func(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Apply a given function to the wrapped coefficients.

        Parameters
        ----------
        func: function(<T>, *args, **kwargs) -> <T>
            Function to be applied to each and every coefficient (data
            array) wrapped by this Vector, that must return a similar
            array (same type, shape and dtype).
        *args and **kwargs to `func` may also be passed.

        Returns
        -------
        vector: Self
            Vector similar to the present one, wrapping the resulting data.
        """
        coefs = {
            key: func(coef, *args, **kwargs)
            for key, coef in self.coefs.items()
        }
        return type(self)(coefs)

    def _apply_operation(
        self,
        other: Any,
        func: Callable[[Any, Any], Any],
    ) -> Self:
        """Apply an operation to combine this vector with another.

        Parameters
        ----------
        other: Vector
            Vector with the same names, shapes and dtypes as this one.
        func: function(<T>, <T>) -> <T>
            Function to be applied to combine the data arrays stored
            in this vector and the `other` one.

        Returns
        -------
        vector: Self
            Vector similar to the present one, wrapping the resulting data.
        """
        # Case when operating on two Vector objects.
        if isinstance(other, tuple(self.compatible_vector_types)):
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
        # Case when the two vectors have incompatible types.
        if isinstance(other, Vector):
            raise TypeError(
                f"Cannot {func.__name__} {type(self).__name__} object with "
                f"a vector of incompatible type {type(other).__name__}."
            )
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
    ) -> Self:
        return self._apply_operation(other, self._op_add)

    def __radd__(
        self,
        other: Any,
    ) -> Self:
        return self.__add__(other)

    def __sub__(
        self,
        other: Any,
    ) -> Self:
        return self._apply_operation(other, self._op_sub)

    def __rsub__(
        self,
        other: Any,
    ) -> Self:
        return -1 * self.__sub__(other)

    def __mul__(
        self,
        other: Any,
    ) -> Self:
        return self._apply_operation(other, self._op_mul)

    def __rmul__(
        self,
        other: Any,
    ) -> Self:
        return self.__mul__(other)

    def __truediv__(
        self,
        other: Any,
    ) -> Self:
        return self._apply_operation(other, self._op_div)

    def __rtruediv__(
        self,
        other: Any,
    ) -> Self:
        return self.__mul__(1 / other)

    def __pow__(
        self,
        other: Any,
    ) -> Self:
        return self._apply_operation(other, self._op_pow)

    @abstractmethod
    def __eq__(
        self,
        other: Any,
    ) -> bool:
        """Equality operator for Vector classes.

        Two Vectors should be deemed equal if they have the same
        specs (same keys, shapes and dtypes) and the same values.

        Otherwise, this magic method should return False.
        """

    @abstractmethod
    def sign(
        self,
    ) -> Self:
        """Return a Vector storing the sign of each coefficient."""

    @abstractmethod
    def minimum(
        self,
        other: Union[Self, float, ArrayLike],
    ) -> Self:
        """Compute coef.-wise, element-wise minimum wrt to another Vector."""

    @abstractmethod
    def maximum(
        self,
        other: Union[Self, float, ArrayLike],
    ) -> Self:
        """Compute coef.-wise, element-wise maximum wrt to another Vector."""

    @abstractmethod
    def sum(
        self,
    ) -> Self:
        """Compute coefficient-wise sum of elements."""


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
