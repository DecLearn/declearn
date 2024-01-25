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

from typing import List, Union

import numpy as np
from declearn.model.api import Vector, VectorSpec

from declearn.secagg.joye_libert._encrypt import ArraySpec
from declearn.secagg.joye_libert._joye_libert import (
    DEFAULT_BIPRIME,
    decrypt_sum,
)
from declearn.secagg.utils import Quantizer

__all__ = [
    "JoyeLibertDecrypter",
]


class JoyeLibertDecrypter:
    """Controller for the decryption of (homomorphic) sums of encrypted values.

    This class makes use of primitives implementing an algorithm
    proposed by Joye & Libert [1] for homomorphic summation.

    References
    ----------
    [1] Joye & Libert, 2013.
        A Scalable Scheme for Privacy-Preserving Aggregation
        of Time-Series Data.
        https://marcjoye.github.io/papers/JL13aggreg.pdf
    """

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
        self.pub_key = pub_key
        self.n_peers = n_peers
        self.biprime = biprime
        self.quantizer = Quantizer(val_range=clipval, int_range=2**bitsize)
        self._qt_corr = (n_peers - 1) * self.quantizer.quantize_value(0.0)
        self._t_index = 0

    def decrypt_int(
        self,
        value: int,
    ) -> int:
        """Decrypt an encrypted sum of private integer values."""
        output = decrypt_sum(
            value,
            index=self._t_index,
            public=self.pub_key,
            modulus=self.biprime,
        )
        self._t_index += 1
        return output

    def decrypt_float(
        self,
        value: int,
    ) -> float:
        """Decrypt an encrypted sum of private float values."""
        int_val = self.decrypt_int(value)
        int_val -= self._qt_corr
        return self.quantizer.unquantize_value(int_val)

    def decrypt_numpy_array(
        self,
        values: List[int],
        specs: ArraySpec,
    ) -> np.ndarray:
        """Decrypt an encrypted sum of private numpy array of values."""
        s_val = [self.decrypt_int(val) for val in values]
        shape, dtype = specs
        if issubclass(np.dtype(dtype).type, np.floating):
            s_val = self.quantizer.unquantize_list(  # type: ignore[assignment]
                [val - self._qt_corr for val in values]
            )
        return np.array(s_val, dtype=dtype).reshape(shape)

    def decrypt_vector(
        self,
        values: List[int],
        specs: VectorSpec,
    ) -> Vector:
        """Decrypt an encrypted sum of private Vector of values."""
        int_val = [self.decrypt_int(val) - self._qt_corr for val in values]
        flt_val = self.quantizer.unquantize_list(int_val)
        return Vector.build_from_specs(flt_val, specs)

    def decrypt_value(
        self,
        values: List[int],
        specs: Union[bool, ArraySpec, VectorSpec],
    ) -> Union[int, float, np.ndarray, Vector]:
        """Decrypt an encrypted sum of values of any supported type.

        Parameters
        ----------
        encrypted:
            List of one or more integers storing the encrypted inputs.
        specs:
            Value indicating specifications of the input value, the
            type of which depends on that of the cleartext value:
                - `VectorSpec` for `Vector` values
                - `(shape, dtype)` tuple for `np.ndarray` values
                - `is_float` bool for scalar int or float values

        Returns
        -------
        value:
            Decrypted data, matching that of the private values
            encrypted and summed into `values`.
        """
        if isinstance(specs, (tuple, list)):
            return self.decrypt_numpy_array(values, specs)
        if isinstance(specs, VectorSpec):
            return self.decrypt_vector(values, specs)
        if isinstance(specs, bool):
            func = self.decrypt_float if specs else self.decrypt_int
            return func(values[0])
        raise TypeError(
            f"Cannot decrypt inputs with specs of type '{type(specs)}'."
        )
