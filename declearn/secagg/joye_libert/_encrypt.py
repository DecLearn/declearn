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

"""Data encrypter for SecAgg using Joye-Libert homomorphic summation."""

from typing import List, Tuple, Union

import numpy as np
from declearn.model.api import Vector, VectorSpec

from declearn.secagg.joye_libert._aggregate import (
    ArraySpec,
    EncryptedSpecs,
    JLSAggregate,
)
from declearn.secagg.joye_libert._joye_libert import (
    DEFAULT_BIPRIME,
    encrypt,
)
from declearn.secagg.utils import Quantizer
from declearn.utils import Aggregate

__all__ = [
    "JoyeLibertEncrypter",
]


class JoyeLibertEncrypter:
    """Controller for the encryption of values that need homomorphic summation.

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
        self.prv_key = prv_key
        self.biprime = biprime
        self.quantizer = Quantizer(
            val_range=clipval, int_range=2**bitsize - 1
        )
        self._t_index = 0

    def encrypt_int(
        self,
        value: int,
    ) -> int:
        """Encrypt a private integer value."""
        output = encrypt(
            value,
            index=self._t_index,
            secret=self.prv_key,
            modulus=self.biprime,
        )
        self._t_index += 1
        return output

    def encrypt_float(
        self,
        value: float,
    ) -> int:
        """Encrypt a private float value."""
        int_val = self.quantizer.quantize_value(value)
        return self.encrypt_int(int_val)

    def encrypt_numpy_array(
        self,
        value: np.ndarray,
    ) -> Tuple[List[int], ArraySpec]:
        """Encrypt a private numpy array of values."""
        if issubclass(value.dtype.type, np.integer):
            int_val = value.flatten().tolist()
        elif issubclass(value.dtype.type, np.floating):
            flt_val = value.flatten().tolist()
            int_val = self.quantizer.quantize_list(flt_val)
        else:
            raise TypeError(
                f"Cannot encrypt numpy array with '{value.dtype}' dtype."
            )
        encrypted = [self.encrypt_int(val) for val in int_val]
        array_spec = (list(value.shape), value.dtype.name)
        return encrypted, array_spec

    def encrypt_vector(
        self,
        value: Vector,
    ) -> Tuple[List[int], VectorSpec]:
        """Encrypt a private Vector of values."""
        flt_val, v_spec = value.flatten()
        int_val = self.quantizer.quantize_list(flt_val)
        enc_val = [self.encrypt_int(val) for val in int_val]
        return enc_val, v_spec

    def encrypt_value(
        self,
        value: Union[int, float, np.ndarray, Vector],
    ) -> Tuple[List[int], Union[bool, ArraySpec, VectorSpec]]:
        """Encrypt a given value of any supported type.

        Returns
        -------
        encrypted:
            List of one or more integers storing the encrypted inputs.
        specs:
            Value indicating specifications of the input value, the
            type of which depends on that of `value`:
                - `VectorSpec` for `Vector` values
                - `(shape, dtype)` tuple for `np.ndarray` values
                - `is_float` bool for scalar int or float values
        """
        if isinstance(value, np.ndarray):
            return self.encrypt_numpy_array(value)
        if isinstance(value, Vector):
            return self.encrypt_vector(value)
        if isinstance(value, float):
            return [self.encrypt_float(value)], True
        if isinstance(value, int):
            return [self.encrypt_int(value)], False
        raise TypeError(f"Cannot encrypt inputs with type '{type(value)}'.")

    def encrypt_aggregate(
        self,
        value: Aggregate,
    ) -> JLSAggregate:
        """Encrypt an 'Aggregate' instance that needs secure aggregation."""
        # Gather fields that need encryption and fields that remain cleartext.
        cryptable, cleartext = value.prepare_for_secagg()
        # Iteratively encrypt fields that need it.
        encrypted = []  # type: List[int]
        enc_specs = []  # type: EncryptedSpecs
        for key, val in cryptable.items():
            enc_v, spec = self.encrypt_value(val)
            encrypted.extend(enc_v)
            enc_specs.append((key, len(enc_v), spec))
        # Wrap the results into a 'JLSAggregate' structure.
        return JLSAggregate(
            encrypted=encrypted,
            enc_specs=enc_specs,
            cleartext=cleartext,
            biprime=self.biprime,
            agg_cls=type(value),
            n_aggrg=1,
        )
