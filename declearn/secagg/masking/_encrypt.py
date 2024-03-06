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

"""Masking-based encrypter for SecAgg."""

import functools
from typing import List, Tuple, TypeVar, Union

import numpy as np

from declearn.model.api import Vector, VectorSpec
from declearn.secagg.masking._aggregate import (
    ArraySpec,
    EncryptedSpecs,
    MaskedAggregate,
)
from declearn.secagg.utils import Quantizer, get_numpy_uint_dtype
from declearn.utils import Aggregate

__all__ = [
    "MaskingEncrypter",
]

AggregateT = TypeVar("AggregateT", bound=Aggregate)


class MaskingEncrypter:
    """Masking-based encrypter for SecAgg."""

    def __init__(
        self,
        pos_masks_seeds: List[int],
        neg_masks_seeds: List[int],
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        self._pos_rng = [
            np.random.default_rng(seed) for seed in pos_masks_seeds
        ]
        self._neg_rng = [
            np.random.default_rng(seed) for seed in neg_masks_seeds
        ]
        self.quantizer = Quantizer(val_range=clipval, int_range=2**bitsize - 1)
        if not self.quantizer.numpy_compatible:
            raise ValueError(
                "'MaskingEncrypter' requires 'bitsize' to be low enough for "
                "compatibility with numpy uint dtypes. This usually means a "
                "bitsize <= 64."
            )
        self.max_int = 2**bitsize
        self._dtype = get_numpy_uint_dtype(self.quantizer.int_range)

    @functools.cached_property
    def n_peers(self) -> int:
        """Number of masking peers."""
        return len(self._pos_rng) + len(self._neg_rng)

    def _generate_masks(
        self,
        n_values: int,
    ) -> np.ndarray:
        """Generate a number of masking values."""
        mask = np.zeros(shape=(n_values,), dtype=self._dtype)
        max_val = self.max_int // self.n_peers
        for rng in self._pos_rng:
            mask += rng.integers(max_val, dtype=self._dtype, size=n_values)
        for rng in self._neg_rng:
            mask -= rng.integers(max_val, dtype=self._dtype, size=n_values)
        return mask

    def encrypt_int(
        self,
        value: int,
    ) -> int:
        mask = int(self._generate_masks(1)[0])
        return value + mask

    def encrypt_float(
        self,
        value: float,
    ) -> int:
        int_val = self.quantizer.quantize_value(value)
        return self.encrypt_int(int_val)

    def encrypt_numpy_array(
        self,
        value: np.ndarray,
    ) -> Tuple[List[int], ArraySpec]:
        if issubclass(value.dtype.type, np.unsignedinteger):
            int_arr = value.flatten()
        elif issubclass(value.dtype.type, (np.integer, np.floating)):
            qnt_arr = self.quantizer.quantize_array(value)
            int_arr = qnt_arr.flatten()
        else:
            raise TypeError(
                f"Cannot encrypt numpy array with '{value.dtype}' dtype."
            )
        msk_arr = self._generate_masks(len(int_arr))
        enc_val = (int_arr + msk_arr).tolist()
        array_spec = (list(value.shape), value.dtype.name)
        return enc_val, array_spec

    def encrypt_vector(
        self,
        value: Vector,
    ) -> Tuple[List[int], VectorSpec]:
        flt_val, v_spec = value.flatten()
        int_arr = np.array(self.quantizer.quantize_list(flt_val))
        msk_arr = self._generate_masks(len(int_arr))
        enc_val = (int_arr + msk_arr).tolist()
        return enc_val, v_spec

    def encrypt_aggregate(
        self,
        value: AggregateT,
    ) -> MaskedAggregate[AggregateT]:
        """Encrypt an 'Aggregate' instance that needs secure aggregation.

        Parameters
        ----------
        value:
            Cleartext `Aggregate`-child-class instance wrapping values
            that need encryption for secure aggregation.

        Returns
        -------
        encrypted:
            `MaskedAggregate` object wrapping encrypted data (and opt. some
            cleartext fields) and specs derived from the input `value`.

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
        encrypted = []  # type: List[int]
        enc_specs = []  # type: EncryptedSpecs
        for key, val in cryptable.items():
            enc_v, spec = self._encrypt_value(val)
            encrypted.extend(enc_v)
            enc_specs.append((key, len(enc_v), spec))
        # Wrap the results into a 'JLSAggregate' structure.
        return MaskedAggregate(
            encrypted=encrypted,
            enc_specs=enc_specs,
            cleartext=cleartext,
            agg_cls=type(value),
            max_int=self.max_int,
            n_aggrg=1,
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
            return [self.encrypt_int(value)], False
        raise TypeError(f"Cannot encrypt inputs with type '{type(value)}'.")
