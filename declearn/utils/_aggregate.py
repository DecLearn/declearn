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

"""Abstract base dataclass for cross-peers data aggregation containers."""

import abc
import dataclasses
from typing import Any, ClassVar, Dict

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.utils._json import add_json_support


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

    By default, fields are aggregated using `default_aggregate`,
    which by default implements the mere summation of two values.
    However, the aggregation rule for any field may be overridden
    by declaring an `aggregate_<field.name>` method.

    Subclasses may also overload the main `aggregate` method, if
    some fields require to be aggregated in a specific way that
    involves crossing values from mutiple ones.

    Serialization
    -------------

    By default, subclass will be made (de)serializable to and from
    JSON, using `declearn.utils.add_json_support` and the `to_dict`
    and `from_dict` methods. This may be prevented by passing the
    `register=False` keyword argument at inheritance time, i.e.:
    `MyAggregate(Aggregate, register=False)`.

    Note that first-child subclasses of `Aggregate` need to define
    the class attribute `_group_key` that acts as a root for their
    children' JSON-registration name.
    """

    _group_key: ClassVar[str]  # Group key for JSON registration.

    def __init_subclass__(
        cls,
        register: bool = True,
    ) -> None:
        """Automatically add JSON support for subclasses."""
        if register:
            name = f"{cls._group_key}>{cls.__name__}"
            add_json_support(
                cls, pack=cls.to_dict, unpack=cls.from_dict, name=name
            )

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
