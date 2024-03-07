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

"""Data decrypter for SecAgg using Joye-Libert homomorphic summation."""

from typing import TypeVar


from declearn.secagg.api import Decrypter, SecureAggregate
from declearn.secagg.masking._aggregate import MaskedAggregate
from declearn.utils import Aggregate

__all__ = [
    "MaskingDecrypter",
]


AggregateT = TypeVar("AggregateT", bound=Aggregate)


class MaskingDecrypter(Decrypter):
    """Controller for the reconstruction of sums of mask-encrypted values.

    TODO: Add references and algorithm details.
    """

    secure_aggregate_cls = MaskedAggregate

    def __init__(
        self,
        n_peers: int,
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        super().__init__(n_peers, bitsize=bitsize, clipval=clipval)
        if not self.quantizer.numpy_compatible:
            raise ValueError(
                "'MaskingDecrypter' requires 'bitsize' to be low enough for "
                "compatibility with numpy uint dtypes. This usually means a "
                "bitsize <= 64."
            )
        self.max_int = 2**bitsize

    def decrypt_uint(
        self,
        value: int,
    ) -> int:
        return value % self.max_int

    def decrypt_aggregate(
        self,
        value: SecureAggregate[AggregateT],
    ) -> AggregateT:
        if isinstance(value, MaskedAggregate):
            if value.max_int != self.max_int:
                raise ValueError(
                    "Cannot decrypt a 'MaskedAggregate' with mismatching "
                    "'max_int'."
                )
        return super().decrypt_aggregate(value)
