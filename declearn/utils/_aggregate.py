# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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

"""Abstract base dataclass for cross-peers data aggregation containers."""

import abc
import dataclasses
from typing import Any, ClassVar, Dict, Optional, Self, Tuple

from declearn.utils._json import add_json_support
from declearn.utils._register import create_types_registry, register_type

__all__ = [
    "Aggregate",
]


@dataclasses.dataclass
class Aggregate(metaclass=abc.ABCMeta):
    """Abstract base dataclass for cross-peers data aggregation containers.

    This class defines an API for containers of values that are
    to be shared across peers and aggregated with other similar
    instances.

    It is typically intended as a base structure to share model
    updates, optimizer auxiliary variables, metadata, analytics
    or model evaluation metrics that are to be aggregated, and
    eventually finalized into some results, across a federated
    or decentralized network of data-holding peers.

    Aggregation
    -----------

    By default, fields are aggregated using `default_aggregate`,
    which by default implements the mere summation of two values.
    However, the aggregation rule for any field may be overridden
    by declaring an `aggregate_<field.name>` method.

    Subclasses may also overload the main `aggregate` method, if
    some fields require to be aggregated in a specific way that
    involves crossing values from mutiple ones.

    Secure Aggregation
    ------------------

    The `prepare_for_secagg` method defines whether an `Aggregate`
    is suitable for secure aggregation, and if so, which fields
    are to be encrypted/sum-decrypted, and which are to be shared
    in cleartext and aggregated similarly as in cleartext mode.

    By default, subclasses are assumed to support secure summation
    and require it for each and every field. The method should be
    overridden when this is not the case, returning a pair of dict
    storing, respectively, fields that require secure summation,
    and fields that are to remain cleartext. If secure aggregation
    is not compatible with the subclass, the method should raise a
    `NotImplementedError`.

    Serialization
    -------------

    By default, subclasses will be made (de)serializable to and from
    JSON, using `declearn.utils.add_json_support` and the `to_dict`
    and `from_dict` methods. They will also be type-registered using
    `declearn.utils.register_type`. This may be prevented by passing
    the `register=False` keyword argument at inheritance time, i.e.
    `class MyAggregate(Aggregate, register=False):`.

    For this to succeed, first-child subclasses of `Aggregate` need
    to define the class attribute `_group_key`, that acts as a root
    for their children' JSON-registration name, and the group name
    for their type registration. They also need to be passed the
    `base_cls=True` keyword argument at inheritance time, i.e.
    `class FirstChild(Aggregate, base_cls=True):`.
    """

    _group_key: ClassVar[str]  # Group key for JSON registration.

    def __init_subclass__(
        cls,
        base_cls: bool = False,
        register: bool = True,
    ) -> None:
        """Automatically type-register and add JSON support for subclasses."""
        if base_cls:
            create_types_registry(cls, name=cls._group_key)
        if register:
            name = f"{cls._group_key}>{cls.__name__}"
            add_json_support(
                cls, pack=cls.to_dict, unpack=cls.from_dict, name=name
            )
            register_type(cls, name=cls.__name__, group=cls._group_key)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this instance."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> Self:
        """Instantiate from an object's dict representation."""
        return cls(**data)

    def __add__(
        self,
        other: Any,
    ) -> Self:
        """Overload the sum operator to aggregate multiple instances."""
        try:
            return self.aggregate(other)
        except TypeError:
            return NotImplemented

    def __radd__(
        self,
        other: Any,
    ) -> Self:
        """Enable `0 + Self -> Self`, to support `sum(Iterator[Self])`."""
        if isinstance(other, int) and not other:
            return self
        return NotImplemented

    def aggregate(
        self,
        other: Self,
    ) -> Self:
        """Aggregate this with another instance of the same class.

        Parameters
        ----------
        other:
            Another instance of the same type as `self`.

        Returns
        -------
        aggregated:
            An instance of the same class containing aggregated values.

        Raises
        ------
        TypeError
            If `other` is of unproper type.
        ValueError
            If any field's aggregation fails.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"'{self.__class__.__name__}.aggregate' received a wrongful "
                f"'other'  argument: excepted same type, got '{type(other)}'."
            )
        # Run the fields' aggregation, wrapping any exception as ValueError.
        try:
            results = {
                field.name: getattr(
                    self, f"aggregate_{field.name}", self.default_aggregate
                )(getattr(self, field.name), getattr(other, field.name))
                for field in dataclasses.fields(self)
            }
        except Exception as exc:
            raise ValueError(
                "Exception encountered while aggregating two instances "
                f"of '{self.__class__.__name__}': {repr(exc)}."
            ) from exc
        # If everything went right, return the resulting AuxVar.
        return self.__class__(**results)

    @staticmethod
    def default_aggregate(
        val_a: Any,
        val_b: Any,
    ) -> Any:
        """Aggregate two values using the default summation operator."""
        return val_a + val_b

    def prepare_for_secagg(
        self,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Return content for secure-aggregation of instances of this class.

        Returns
        -------
        secagg_fields:
            Dict storing fields that are compatible with encryption
            and secure aggregation using mere summation.
        clrtxt_fields:
            Dict storing fields that are to be shared in cleartext
            version. They will be aggregated using the same method
            as usual (`aggregate_<name>` or `default_aggregate`).

        Raises
        ------
        NotImplementedError
            If this class does not support Secure Aggregation,
            and its contents should therefore not be shared.

        Notes for developers
        --------------------
        - `secagg_fields` values should have one of the following types:
            - `int` (for positive integer values only)
            - `float`
            - `numpy.ndarray` (with any floating or integer dtype)
            - `Vector`
        - Classes that are incompatible with secure aggregation should
          implement a `raise NotImplementedError` statement, explaining
          whether SecAgg cannot or is yet-to-be supported.
        """
        return self.to_dict(), None
