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

"""Abstract base class for decryption controllers for SecAgg."""

import abc
from typing import Any, ClassVar, Dict, List, Type, TypeVar, Union

import numpy as np

from declearn.model.api import Vector, VectorSpec
from declearn.secagg.api._aggregate import ArraySpec, SecureAggregate
from declearn.secagg.utils import Quantizer
from declearn.utils import Aggregate

__all__ = [
    "Decrypter",
]


AggregateT = TypeVar("AggregateT", bound=Aggregate)


class Decrypter(metaclass=abc.ABCMeta):
    """ABC controller for the decryption of summed encrypted values."""

    secure_aggregate_cls: ClassVar[Type[SecureAggregate]]

    def __init__(
        self,
        n_peers: int,
        /,
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        """Instantiate the SecAgg decryption controller.

        Parameters
        ----------
        n_peers:
            Number of peers that contribute values to the sums.
            This is required to prevent unquantization errors
            when decrypting float values.
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
        self.n_peers = n_peers
        self.quantizer = Quantizer(val_range=clipval, int_range=2**bitsize - 1)
        self._qt_corr = (n_peers - 1) * self.quantizer.quantize_value(0.0)

    @abc.abstractmethod
    def sum_encrypted(
        self,
        values: List[int],
    ) -> int:
        """Sum some encrypted integer values into a single one.

        Parameters
        ----------
        values:
            Encrypted values that need summation with each other.

        Returns
        -------
        val:
            Encrypted value resulting from the input ones' aggregation.
        """

    @abc.abstractmethod
    def decrypt_uint(
        self,
        value: int,
    ) -> int:
        """Decrypt an encrypted sum of private positive integer values.

        Parameters
        ----------
        value:
            Encrypted sum of private positive integer values.

        Returns
        -------
        decrypted:
            Decrypted positive integer value.
        """

    def _correct_quantized_sum(
        self,
        value: int,
    ) -> int:
        """Apply some correction to a quantized cleartext sum of values.

        This method subtracts `(n - 1) * q(0)` from inputs, to account
        for the quantizer shifting inputs by the clipping value.
        """
        return value - self._qt_corr

    def decrypt_float(
        self,
        value: int,
    ) -> float:
        """Decrypt an encrypted sum of private float values.

        Parameters
        ----------
        value:
            Encrypted sum of private float values.

        Returns
        -------
        decrypted:
            Decrypted float value.
        """
        int_val = self.decrypt_uint(value)
        int_val = self._correct_quantized_sum(int_val)
        return self.quantizer.unquantize_value(int_val)

    def decrypt_numpy_array(
        self,
        values: List[int],
        specs: ArraySpec,
    ) -> np.ndarray:
        """Decrypt an encrypted sum of private numpy array of values.

        Parameters
        ----------
        values:
            List of encrypted sums of private array values.
        specs:
            Tuple storing array shape and dtype metadata.

        Returns
        -------
        decrypted:
            Decrypted numpy array instance, with shape and dtype
            matching input `specs`.
        """
        s_val = [self.decrypt_uint(val) for val in values]
        shape, dtype = specs
        if not issubclass(np.dtype(dtype).type, np.unsignedinteger):
            s_val = [self._correct_quantized_sum(val) for val in s_val]
            s_val = self.quantizer.unquantize_list(s_val)  # type: ignore
            if issubclass(np.dtype(dtype).type, np.signedinteger):
                s_val = [round(x) for x in s_val]
        return np.array(s_val, dtype=dtype).reshape(shape)

    def decrypt_vector(
        self,
        values: List[int],
        specs: VectorSpec,
    ) -> Vector:
        """Decrypt an encrypted sum of private Vector of values.

        Parameters
        ----------
        values:
            List of encrypted sums of private vector coefficient values.
        specs:
            `VectorSpec` instance storing metadata of the `Vector`
            structure.

        Returns
        -------
        decrypted:
            Decrypted `Vector` instance, with specs matching `specs`.
        """
        int_val = [self.decrypt_uint(val) for val in values]
        int_val = [self._correct_quantized_sum(val) for val in int_val]
        flt_val = self.quantizer.unquantize_list(int_val)
        return Vector.build_from_specs(flt_val, specs)

    def decrypt_aggregate(
        self,
        value: SecureAggregate[AggregateT],
    ) -> AggregateT:
        """Decrypt a 'SecureAggregate' wrapping a summation of private values.

        Parameters
        ----------
        value:
            `SecureAggregate` object wrapping aggregated encrypted data
            and associate metadata about its source `Aggregate` type.
            The exact expected type depends on the controller's type,
            and indicated by the `secure_aggregate_cls` class attribute.

        Returns
        -------
        decrypted:
            `Aggregate` instance recovered from `value`, with cleartext
            fields storing securely aggregated values.
        """
        # Perform basic verifications.
        if not isinstance(value, SecureAggregate):
            raise TypeError(
                f"'{self.__class__.__name__}.decrypt_aggregate' expects "
                f"'{self.secure_aggregate_cls.__name__}' inputs but received "
                f"a '{type(value)}'."
            )
        if value.n_aggrg != self.n_peers:
            raise ValueError(
                f"'{self.__class__.__name__}.decrypt_aggregate' expects "
                f"input '{self.secure_aggregate_cls.__name__}' to result "
                f"from the summation of {self.n_peers} instances, but it "
                f"appears {value.n_aggrg} values were in fact summed."
            )
        # Iteratively decrypt and recover encrypted fields.
        srt = end = 0
        fields: Dict[str, Any] = {}
        for name, size, specs in value.enc_specs:
            end += size
            fields[name] = self._decrypt_value(value.encrypted[srt:end], specs)
            srt = end
        # Instantiate and return from decrypted and cleartext fields.
        return value.agg_cls(**fields, **value.cleartext)

    def _decrypt_value(
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
            func = self.decrypt_float if specs else self.decrypt_uint
            return func(values[0])
        raise TypeError(
            f"Cannot decrypt inputs with specs of type '{type(specs)}'."
        )
