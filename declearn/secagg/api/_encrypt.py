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

"""Abstract base class for encryption controllers for SecAgg."""

import abc
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np

from declearn.model.api import Vector, VectorSpec
from declearn.secagg.api._aggregate import (
    ArraySpec,
    EncryptedSpecs,
    SecureAggregate,
)
from declearn.secagg.utils import Quantizer
from declearn.utils import Aggregate

__all__ = [
    "Encrypter",
]


AggregateT = TypeVar("AggregateT", bound=Aggregate)


class Encrypter(metaclass=abc.ABCMeta):
    """ABC controller for the encryption of values that need summation."""

    def __init__(
        self,
        /,
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        """Instantiate the SecAgg encryption controller.

        Parameters
        ----------
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
        self.quantizer = Quantizer(val_range=clipval, int_range=2**bitsize - 1)

    @abc.abstractmethod
    def encrypt_uint(
        self,
        value: int,
    ) -> int:
        """Encrypt a private positive integer value.

        Notes
        -----
        If this value, or any value it is meant to be combined with, may
        be negative, you should use `encrypt_float(float(value))` instead
        of this method. Decryption will then need to be done as though it
        were a float value, with further rounding of the output back into
        an int.

        You may alternatively wrap signed int values into numpy arrays
        and use `encrypt_array`, to lower the type handling burden.

        Parameters
        ----------
        value:
            Cleartext positive integer value to encrypt.

        Returns
        -------
        encrypted:
            Encrypted value, as a (possibly-large) integer.
        """

    def encrypt_float(
        self,
        value: float,
    ) -> int:
        """Encrypt a private float value.

        Parameters
        ----------
        value:
            Cleartext float value to encrypt.

        Returns
        -------
        encrypted:
            Encrypted value, as a (possibly-large) integer.
        """
        int_val = self.quantizer.quantize_value(value)
        return self.encrypt_uint(int_val)

    def encrypt_numpy_array(
        self,
        value: np.ndarray,
    ) -> Tuple[List[int], ArraySpec]:
        """Encrypt a private numpy array of values.

        Parameters
        ----------
        value:
            Cleartext numpy array storing numerical values.
            This array may have a float, uint or int dtype.

        Returns
        -------
        encrypted:
            Encrypted values, as a list of large integers.
        specs:
            Tuple storing array shape and dtype metadata,
            in cleartext and JSON-serializable format.

        Raises
        ------
        TypeError
            If `value` as neither an integer nor floating dtype.
        """
        if issubclass(value.dtype.type, np.unsignedinteger):
            int_val = value.flatten().tolist()
        elif issubclass(value.dtype.type, (np.integer, np.floating)):
            if self.quantizer.numpy_compatible:
                qnt_arr = self.quantizer.quantize_array(value)
                int_val = qnt_arr.flatten().tolist()
            else:
                flt_val = value.flatten().tolist()
                int_val = self.quantizer.quantize_list(flt_val)
        else:
            raise TypeError(
                f"Cannot encrypt numpy array with '{value.dtype}' dtype."
            )
        encrypted = [self.encrypt_uint(val) for val in int_val]
        array_spec = (list(value.shape), value.dtype.name)
        return encrypted, array_spec

    def encrypt_vector(
        self,
        value: Vector,
    ) -> Tuple[List[int], VectorSpec]:
        """Encrypt a private Vector of values.

        Parameters
        ----------
        value:
            Cleartext declearn `Vector` storing numerical values.

        Returns
        -------
        encrypted:
            Encrypted values, as a list of large integers.
        specs:
            VectorSpec associated with `value`.
        """
        flt_val, v_spec = value.flatten()
        int_val = self.quantizer.quantize_list(flt_val)
        enc_val = [self.encrypt_uint(val) for val in int_val]
        return enc_val, v_spec

    def encrypt_aggregate(
        self,
        value: AggregateT,
    ) -> SecureAggregate[AggregateT]:
        """Encrypt an 'Aggregate' instance that needs secure aggregation.

        Parameters
        ----------
        value:
            Cleartext `Aggregate`-child-class instance wrapping values
            that need encryption for secure aggregation.

        Returns
        -------
        encrypted:
            `SecureAggregate` object wrapping encrypted data (and opt. some
            cleartext fields) and specs derived from the input `value`. The
            exact type of the output depends on the controller's type.

        Raises
        ------
        NotImplementedError
            If the input `Aggregate` type does not support secure aggregation.
        TypeError
            If any field marked as requiring secure aggregation is not a
            positive int, float, numerical numpy array or declearn Vector
            instance.
        """
        # Gather fields that need encryption and fields that remain cleartext.
        cryptable, cleartext = value.prepare_for_secagg()
        # Iteratively encrypt fields that need it.
        encrypted: List[int] = []
        enc_specs: EncryptedSpecs = []
        for key, val in cryptable.items():
            enc_v, spec = self._encrypt_value(val)
            encrypted.extend(enc_v)
            enc_specs.append((key, len(enc_v), spec))
        # Wrap the results into a 'JLSAggregate' structure.
        return self.wrap_into_secure_aggregate(
            encrypted=encrypted,
            enc_specs=enc_specs,
            cleartext=cleartext,
            agg_cls=type(value),
        )

    def _encrypt_value(
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
            return [self.encrypt_uint(value)], False
        raise TypeError(f"Cannot encrypt inputs with type '{type(value)}'.")

    @abc.abstractmethod
    def wrap_into_secure_aggregate(
        self,
        encrypted: List[int],
        enc_specs: EncryptedSpecs,
        cleartext: Optional[Dict[str, Any]],
        agg_cls: Type[AggregateT],
    ) -> SecureAggregate[AggregateT]:
        """Wrap up an aggregate's encrypted values and metadata."""
