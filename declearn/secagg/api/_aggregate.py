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

"""Abstract base class to wrap up encrypted 'Aggregate' objects for SecAgg."""

import abc
import copy
from typing import (  # fmt: off
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from declearn.model.api import VectorSpec
from declearn.utils import (
    Aggregate,
    access_registered,
    access_registration_info,
    add_json_support,
)

__all__ = [
    "ArraySpec",
    "EncryptedSpecs",
    "SecureAggregate",
]

AggregateT = TypeVar("AggregateT", bound=Aggregate)

ArraySpec = Tuple[List[int], str]
"""Type-hint alias for specifications of an encrypted numpy array.

An `ArraySpec` is merely the shape and dtype of the original array.
"""

EncryptedSpecs = List[Tuple[str, int, Union[bool, ArraySpec, VectorSpec]]]
"""Type-hint alias for specifications of encrypted 'Aggregate' fields.

An `EncryptedSpecs` is a list of tuples that each specify a given field, as:

- a string (the field's name)
- an integer (the number of scalar values in the field)
- a specifier that depends on the field's type:
    - a `bool` for uint (`False`) or float (`True`) fields;
    - a `(shape, dtype)` tuple (`ArraySpec`) for numpy array fields;
    - a `VectorSpec` for declearn Vector fields.
"""


class SecureAggregate(Generic[AggregateT], metaclass=abc.ABCMeta):
    """Abstract 'Aggregate'-like wrapper for encrypted 'Aggregate' objects."""

    def __init__(
        self,
        encrypted: List[int],
        enc_specs: EncryptedSpecs,
        cleartext: Optional[Dict[str, Any]],
        agg_cls: Type[AggregateT],
        n_aggrg: int,
    ) -> None:
        """Instantiate a SecureAggregate.

        Parameters
        ----------
        encrypted:
            List of encrypted values that need sum-aggregation.
        enc_specs:
            Source specifications of encrypted values, as a list of
            tuples denoting `(name, number_of_values, optional_specs)`.
        cleartext:
            Optional dict storing some cleartext fields that do not
            require encryption.
        agg_cls:
            Type of the original `Aggregate` that was encrypted into
            this instance.
        n_aggrg:
            Number of individual encrypted aggregated having been
            aggregated into this instance.
        """
        # backend class; pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        self.encrypted = encrypted
        self.enc_specs = enc_specs
        self.cleartext = cleartext or {}
        self.agg_cls = agg_cls
        self.n_aggrg = n_aggrg

    def __init_subclass__(
        cls,
        register: bool = True,
    ) -> None:
        """Automatically add JSON support for subclasses."""
        add_json_support(
            cls=cls, pack=cls.to_dict, unpack=cls.from_dict, name=cls.__name__
        )

    def aggregate(
        self,
        other: Self,
    ) -> Self:
        """Aggregate this with another instance of matching specs."""
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"'{self.__class__.__name__}.aggregate' expects an input "
                f"with the same type, but received '{type(other)}'."
            )
        if self.enc_specs != other.enc_specs:
            raise ValueError(
                f"Cannot sum '{self.__class__.__name__}' instances with"
                " distinct specs for encrypted values."
            )
        encrypted = self.aggregate_encrypted(self.encrypted, other.encrypted)
        default = self.agg_cls.default_aggregate
        cleartext = (
            None
            if self.cleartext is None
            else {
                key: getattr(self.agg_cls, f"aggregate_{key}", default)(
                    val, other.cleartext[key]
                )
                for key, val in self.cleartext.items()
            }
        )
        n_aggrg = self.n_aggrg + other.n_aggrg
        return self.__class__(
            encrypted=encrypted,
            enc_specs=self.enc_specs,
            cleartext=cleartext,
            agg_cls=self.agg_cls,
            n_aggrg=n_aggrg,
        )

    @abc.abstractmethod
    def aggregate_encrypted(
        self,
        val_a: List[int],
        val_b: List[int],
    ) -> List[int]:
        """Aggregate encrypted integer values."""

    def __add__(
        self,
        other: Self,
    ) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.aggregate(other)

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Return a dict representation of this instance.

        Returns
        -------
        data:
            Dict representation of this instance.
        """
        return {
            "encrypted": self.encrypted,
            "enc_specs": self.enc_specs,
            "cleartext": self.cleartext,
            "agg_cls": access_registration_info(self.agg_cls),
            "n_aggrg": self.n_aggrg,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> Self:
        """Instantiate from a dict representation.

        Parameters
        ----------
        data:
            Dict representation, as emitted by this class's `to_dict`.

        Raises
        ------
        TypeError
            If any required key is missing or has improper type or value.
        """
        data = copy.copy(data)
        try:
            data["enc_specs"] = [tuple(s) for s in data["enc_specs"]]
            data["agg_cls"] = access_registered(*data["agg_cls"])
            return cls(**data)
        except Exception as exc:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}' from input dict: "
                f"raised '{repr(exc)}'."
            ) from exc
