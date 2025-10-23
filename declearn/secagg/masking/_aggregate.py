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

from declearn.secagg.api import EncryptedSpecs, SecureAggregate
from declearn.utils import Aggregate

__all__ = [
    "MaskedAggregate",
]

AggregateT = TypeVar("AggregateT", bound=Aggregate)


class MaskedAggregate(SecureAggregate[AggregateT]):
    """'Aggregate'-like container for mask-encrypted 'Aggregate' objects."""

    # pylint: disable-next=too-many-positional-arguments
    def __init__(  # noqa: PLR0913
        self,
        encrypted: List[int],
        enc_specs: EncryptedSpecs,
        cleartext: Optional[Dict[str, Any]],
        agg_cls: Type[AggregateT],
        max_int: int = 2**64,
        n_aggrg: int = 1,
    ) -> None:
        """Instantiate a MaskedAggregate.

        Parameters
        ----------
        encrypted:
            List of masked values that need aggregation.
        enc_specs:
            Source specifications of encrypted values, as a list of
            tuples denoting `(name, number_of_values, optional_specs)`.
        cleartext:
            Optional dict storing some cleartext fields that do not
            require encryption.
        agg_cls:
            Type of the original `Aggregate` that was encrypted into
            this instance.
        max_int:
            Integer defining a positive integer field for quantized
            and masked values.
        n_aggrg:
            Number of individual encrypted aggregated having been
            aggregated into this instance.
        """
        # backend class; pylint: disable=too-many-arguments
        super().__init__(encrypted, enc_specs, cleartext, agg_cls, n_aggrg)
        self.max_int = max_int

    def aggregate(
        self,
        other: Self,
    ) -> Self:
        """Aggregate this with another instance of matching specs."""
        if isinstance(other, self.__class__) and self.max_int != other.max_int:
            raise ValueError(
                f"Cannot sum '{self.__class__.__name__}' instances with"
                " distinct field-defining maximum integer values."
            )
        output = super().aggregate(other)
        output.max_int = self.max_int
        return output

    def aggregate_encrypted(
        self,
        val_a: List[int],
        val_b: List[int],
    ) -> List[int]:
        return [
            (a + b) % self.max_int for a, b in zip(val_a, val_b, strict=False)
        ]

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
        data["max_int"] = self.max_int
        return data
