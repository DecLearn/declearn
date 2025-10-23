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

"""Secure Aggregation Controller using Joye-Libert homomorphic summation."""

from typing import Any, Dict, List, Optional, Self, Type, TypeVar

from declearn.secagg.api import ArraySpec, EncryptedSpecs, SecureAggregate
from declearn.secagg.joye_libert._primitives import (
    DEFAULT_BIPRIME,
    sum_encrypted,
)
from declearn.utils import (
    Aggregate,
)

__all__ = [
    "ArraySpec",
    "EncryptedSpecs",
    "JLSAggregate",
]

AggregateT = TypeVar("AggregateT", bound=Aggregate)


class JLSAggregate(SecureAggregate[AggregateT]):
    """'Aggregate'-like container for Joye-Libert encrypted values."""

    # pylint: disable-next=too-many-positional-arguments
    def __init__(  # noqa: PLR0913
        self,
        encrypted: List[int],
        enc_specs: EncryptedSpecs,
        cleartext: Optional[Dict[str, Any]],
        agg_cls: Type[AggregateT],
        biprime: int = DEFAULT_BIPRIME,
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
        super().__init__(encrypted, enc_specs, cleartext, agg_cls, n_aggrg)
        self.biprime = biprime

    def aggregate_encrypted(
        self,
        val_a: List[int],
        val_b: List[int],
    ) -> List[int]:
        """Aggregate encrypted integer values."""
        return [
            sum_encrypted(values) for values in zip(val_a, val_b, strict=False)
        ]

    def aggregate(
        self,
        other: Self,
    ) -> Self:
        """Aggregate this with another instance of matching specs."""
        if isinstance(other, self.__class__):
            if self.biprime != other.biprime:
                raise ValueError(
                    f"Cannot sum '{self.__class__.__name__}' instances with"
                    " distinct field-defining biprime modulus."
                )
        output = super().aggregate(other)
        output.biprime = self.biprime
        return output

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Return a dict representation of this instance.

        Returns
        -------
        data:
            Dict representation of this instance.
        """
        data = super().to_dict()
        data["biprime"] = self.biprime
        return data
