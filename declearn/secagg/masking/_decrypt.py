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

"""Data decrypter for SecAgg using Joye-Libert homomorphic summation."""

import math
from typing import List, TypeVar

from declearn.secagg.api import Decrypter, SecureAggregate
from declearn.secagg.masking._aggregate import MaskedAggregate
from declearn.utils import Aggregate

__all__ = [
    "MaskingDecrypter",
]


AggregateT = TypeVar("AggregateT", bound=Aggregate)


class MaskingDecrypter(Decrypter):
    """Controller for the reconstruction of sums of mask-encrypted values.

    This class expects aggregated values to have been uint-quantized and
    masked with values that cancel out on summation. As a consequence it
    merely has the charge to unquantize received sums, while security-
    related efforts are left to the encrypters.

    Contrary to the algorithm from Bonawitz et al. [1] on which it is
    loosely based, this class does not implement mechanisms to support
    clients dropping between setup and decryption. As such, it does not
    require to query clients (possibly maliciously) for anything else
    than the sum of encrypted (i.e. masked) values they emitted.

    References
    ----------
    [1] Bonawitz et al., 2016.
    Practical Secure Aggregation for Federated Learning
    on User-Held Data.
    https://arxiv.org/abs/1611.04482
    """

    secure_aggregate_cls = MaskedAggregate

    def __init__(
        self,
        n_peers: int,
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        """Instantiate a masking-based decryption controller.

        Parameters
        ----------
        n_peers:
            Number of peers that contribute values to the sums.
            This is required to prevent unquantization errors
            when decrypting float values.
        bitsize:
            Maximum bitsize of masked private values.
            This also affects the precision of (un)quantization.
        clipval:
            Maximum absolute value beyond which to clip private
            float values upon quantizing them. This impacts the
            information loss due to (un)quantization of floats.
        """
        # Record encrypted values' bitsize and adjust quantization size.
        # We want sum_{i=1}^n(q(x_i)) < 2**b, hence q(x) < (2**b) / n,
        # which is (less-tightedly) bounded by 2**(b - ceil(log2(n)).
        self.max_int = 2**bitsize
        quant_b = bitsize - int(math.ceil(math.log2(n_peers)))
        super().__init__(n_peers, bitsize=quant_b, clipval=clipval)

    def sum_encrypted(
        self,
        values: List[int],
    ) -> int:
        return sum(values) % self.max_int

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
