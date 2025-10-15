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

from typing import List, TypeVar

from declearn.secagg.api import Decrypter, SecureAggregate
from declearn.secagg.joye_libert._aggregate import JLSAggregate
from declearn.secagg.joye_libert._primitives import (
    DEFAULT_BIPRIME,
    decrypt_sum,
    sum_encrypted,
)
from declearn.utils import Aggregate

__all__ = [
    "JoyeLibertDecrypter",
]


AggregateT = TypeVar("AggregateT", bound=Aggregate)


class JoyeLibertDecrypter(Decrypter):
    """Controller for the decryption of summed Joye-Libert-encrypted values.

    This class makes use of primitives implementing an algorithm
    proposed by Joye & Libert [1] for homomorphic summation.

    It is designed to be used together with the `JoyeLibertEncrypter`
    counterpart class, as well as either the `sum_encrypted` function
    or built-in aggregation rules of `JLSAggregate` to sum encrypted
    values prior to their being input for decryption.

    References
    ----------
    [1] Joye & Libert, 2013.
        A Scalable Scheme for Privacy-Preserving Aggregation
        of Time-Series Data.
        https://marcjoye.github.io/papers/JL13aggreg.pdf
    """

    secure_aggregate_cls = JLSAggregate

    # pylint: disable-next=too-many-positional-arguments
    def __init__(
        self,
        pub_key: int,
        n_peers: int,
        biprime: int = DEFAULT_BIPRIME,
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        """Instantiate the Joye-Libert SecAgg decryption utility.

        Parameters
        ----------
        pub_key:
            Public key used for the decryption of summed values,
            equal to the opposite of the sum of private keys used
            for encryption of the summed values.
        n_peers:
            Number of peers that contribute values to the sums.
            This is required to prevent unquantization errors
            when decrypting float values.
        biprime:
            Public large biprime number defining the modulus for
            Joye-Libert operations. It should be larger than any
            sum of cleartext private values. Private keys should
            have twice its bitsize.
        bitsize:
            Maximum bitsize of quantized cleartext private values.
            The higher the less information lost in quantization
            of float values. Note that any sum of quantized values
            should remain below `biprime` for results correctness.
        clipval:
            Maximum absolute value beyond which to clip private
            float values upon quantizing them. This impacts the
            information loss due to (un)quantization of floats.
        """
        # all arguments are required; pylint: disable=too-many-arguments
        super().__init__(n_peers, bitsize=bitsize, clipval=clipval)
        self.pub_key = pub_key
        self.biprime = biprime
        self._qt_corr = (n_peers - 1) * self.quantizer.quantize_value(0.0)
        self._t_index = 0

    def sum_encrypted(
        self,
        values: List[int],
    ) -> int:
        return sum_encrypted(values, modulus=self.biprime)

    def decrypt_uint(
        self,
        value: int,
    ) -> int:
        output = decrypt_sum(
            value,
            index=self._t_index,
            public=self.pub_key,
            modulus=self.biprime,
        )
        self._t_index += 1
        return output

    def decrypt_aggregate(
        self,
        value: SecureAggregate[AggregateT],
    ) -> AggregateT:
        """Decrypt a 'JLSAggregate' wrapping a summation of private values.

        Parameters
        ----------
        value:
            `JLSAggregate` object wrapping aggregated encrypted data
            and associate metadata about its source `Aggregate` type.

        Returns
        -------
        decrypted:
            `Aggregate` instance recovered from `value`, with cleartext
            fields storing securely aggregated values.
        """
        if isinstance(value, JLSAggregate) and (value.biprime != self.biprime):
            raise ValueError(
                "Cannot decrypt a 'JLSAggregate' with mismatching 'biprime'."
            )
        return super().decrypt_aggregate(value)
