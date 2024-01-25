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

"""Secure Aggregation Controller using Joye-Libert homomorphic summation."""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.model.api import VectorSpec
from declearn.secagg.joye_libert._joye_libert import sum_encrypted
from declearn.utils import (
    Aggregate,
    access_registered,
    access_registration_info,
    add_json_support,
)

__all__ = [
    "ArraySpec",
    "EncryptedSpecs",
    "JLSAggregate",
]

ArraySpec = Tuple[List[int], str]
EncryptedSpecs = List[Tuple[str, int, Union[bool, ArraySpec, VectorSpec]]]


class JLSAggregate:
    """'Aggregate'-like container for Joye-Libert encrypted values."""

    def __init__(
        self,
        encrypted: List[int],
        enc_specs: EncryptedSpecs,
        cleartext: Optional[Dict[str, Any]],
        agg_cls: Type[Aggregate],
        biprime: int,
        n_aggrg: int = 1,
    ) -> None:
        """Instantiate a JLSAggregate.

        Parameters
        ----------
        encrypted:
            List of encrypted values that need aggregation.
        enc_specs:
            Source specifications of encrypted values, as a list of
            tuples denoting `(name, number_of_values, optional_specs)`.
        cleartext:
            Optional dict storing some cleartext fields that do not
            require encryption.
        agg_cls:
            Type of the original `Aggregate` that was encrypted into
            this instance.
        biprime:
            Public biprime number defining the finite integer field
            to which encrypted values pertain.
        n_aggrg:
            Number of individual encrypted aggregated having been
            aggregated into this instance.
        """
        # backend class; pylint: disable=too-many-arguments
        self.encrypted = encrypted
        self.enc_specs = enc_specs
        self.cleartext = cleartext or {}
        self.agg_cls = agg_cls
        self.biprime = biprime
        self.n_aggrg = n_aggrg

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
        if self.biprime != other.biprime:
            raise ValueError(
                f"Cannot sum '{self.__class__.__name__}' instances with"
                " distinct field-defining biprime modulus."
            )
        if self.enc_specs != other.enc_specs:
            raise ValueError(
                f"Cannot sum '{self.__class__.__name__}' instances with"
                " distinct specs for encrypted values."
            )
        encrypted = [
            sum_encrypted(values, self.biprime)
            for values in zip(self.encrypted, other.encrypted)
        ]
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
            biprime=self.biprime,
            n_aggrg=n_aggrg,
        )

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
            "biprime": self.biprime,
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
        try:
            return cls(
                encrypted=data["encrypted"],
                enc_specs=[tuple(s) for s in data["enc_specs"]],
                cleartext=data["cleartext"],
                agg_cls=access_registered(*data["agg_cls"]),
                biprime=data["biprime"],
                n_aggrg=data["n_aggrg"],
            )
        except Exception as exc:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}' from input dict: "
                f"raised '{repr(exc)}'."
            ) from exc


add_json_support(
    cls=JLSAggregate,
    pack=JLSAggregate.to_dict,
    unpack=JLSAggregate.from_dict,
    name="JLSAggregate",
)
