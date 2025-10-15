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

"""Data encrypter for SecAgg using Joye-Libert homomorphic summation."""

from typing import Any, Dict, List, Optional, Type, TypeVar

from declearn.secagg.api import EncryptedSpecs, Encrypter
from declearn.secagg.joye_libert._aggregate import JLSAggregate
from declearn.secagg.joye_libert._primitives import (
    DEFAULT_BIPRIME,
    encrypt,
)
from declearn.utils import Aggregate

__all__ = [
    "JoyeLibertEncrypter",
]


AggregateT = TypeVar("AggregateT", bound=Aggregate)


class JoyeLibertEncrypter(Encrypter):
    """Controller for the Joye-Libert encryption of values that need summation.

    This class makes use of primitives implementing an algorithm
    proposed by Joye & Libert [1] for homomorphic summation.

    It is designed to be used together with the `JoyeLibertDecrypter`
    counterpart class, as well as either the `sum_encrypted` function
    or built-in aggregation rules of `JLSAggregate` to sum its output
    encrypted values prior to their sum's decryption.

    References
    ----------
    [1] Joye & Libert, 2013.
        A Scalable Scheme for Privacy-Preserving Aggregation
        of Time-Series Data.
        https://marcjoye.github.io/papers/JL13aggreg.pdf
    """

    def __init__(
        self,
        prv_key: int,
        biprime: int = DEFAULT_BIPRIME,
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        """Instantiate the Joye-Libert SecAgg encryption utility.

        Parameters
        ----------
        prv_key:
            Private key used for the encryption of values.
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
        super().__init__(bitsize=bitsize, clipval=clipval)
        self.prv_key = prv_key
        self.biprime = biprime
        self._t_index = 0

    def encrypt_uint(
        self,
        value: int,
    ) -> int:
        output = encrypt(
            value,
            index=self._t_index,
            secret=self.prv_key,
            modulus=self.biprime,
        )
        self._t_index += 1
        return output

    def wrap_into_secure_aggregate(
        self,
        encrypted: List[int],
        enc_specs: EncryptedSpecs,
        cleartext: Optional[Dict[str, Any]],
        agg_cls: Type[AggregateT],
    ) -> JLSAggregate[AggregateT]:
        return JLSAggregate(
            encrypted=encrypted,
            enc_specs=enc_specs,
            cleartext=cleartext,
            agg_cls=agg_cls,
            biprime=self.biprime,
            n_aggrg=1,
        )
