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

"""Masking-based encrypter for SecAgg."""

import math
from typing import Any, Dict, List, Optional, Type, TypeVar

import numpy as np

from declearn.secagg.api import EncryptedSpecs, Encrypter
from declearn.secagg.masking._aggregate import MaskedAggregate
from declearn.secagg.utils import get_numpy_uint_dtype
from declearn.utils import Aggregate

__all__ = [
    "MaskingEncrypter",
]

AggregateT = TypeVar("AggregateT", bound=Aggregate)


class MaskingEncrypter(Encrypter):
    """Controller for the mask-based encryption of values that need summation.

    This class makes use of pairwise secret RNG seeds to generate masks
    for (uint-quantized) values that need secure summation, that cancel
    out on summation to recover the (quantized) sum of cleartext values.

    It is based on the approach proposed by Bonawitz et al. [1], without
    thresholding (i.e. without support for clients dropping).

    References
    ----------
    [1] Bonawitz et al., 2016.
    Practical Secure Aggregation for Federated Learning
    on User-Held Data.
    https://arxiv.org/abs/1611.04482
    """

    def __init__(
        self,
        pos_masks_seeds: List[int],
        neg_masks_seeds: List[int],
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        """Instantiate a masking-based encryption controller.

        Parameters
        ----------
        pos_masks_seeds:
            Secret RNG seeds for positive masks generator.
        neg_masks_seeds:
            Secret RNG seeds for negative masks generator.
        bitsize:
            Maximum bitsize of masked private values.
            The higher the less information lost in quantization
            of float values. The quantization domain is reduced
            based on the number of peers (inferred from number
            of mask seeds).
        clipval:
            Maximum absolute value beyond which to clip private
            float values upon quantizing them. This impacts the
            information loss due to (un)quantization of floats.
        """
        # Set up random number generators from input seeds.
        self._pos_rng = [
            np.random.default_rng(seed) for seed in pos_masks_seeds
        ]
        self._neg_rng = [
            np.random.default_rng(seed) for seed in neg_masks_seeds
        ]
        # Record encrypted values' bitsize and adjust quantization size.
        # We want sum_{i=1}^n(q(x_i)) < 2**b, hence q(x) < (2**b) / n,
        # which is (less-tightedly) bounded by 2**(b - ceil(log2(n)).
        self.max_int = 2**bitsize
        n_peers = len(self._pos_rng) + len(self._neg_rng) + 1
        quant_b = bitsize - int(math.ceil(math.log2(n_peers)))
        super().__init__(bitsize=quant_b, clipval=clipval)
        # Identify numpy dtype for masks, if any is large enough.
        self._dtype: Optional[np.dtype] = None
        try:
            self._dtype = get_numpy_uint_dtype(self.max_int - 1)
        except ValueError:
            self._generate_masks = self._generate_masks_large
        else:
            self._generate_masks = self._generate_masks_numpy

    def _generate_masks_large(
        self,
        n_values: int,
    ) -> np.ndarray:
        """Generate a number of masking values."""
        bits_per_mask = int(math.log2(self.max_int)) // 8
        values = [0] * n_values
        for rng in self._pos_rng:
            rbytes = rng.bytes(bits_per_mask * n_values)
            for i in range(n_values):
                dat = rbytes[i * bits_per_mask : (i + 1) * bits_per_mask]
                values[i] = values[i] + int.from_bytes(dat, "big")
        for rng in self._neg_rng:
            rbytes = rng.bytes(bits_per_mask * n_values)
            for i in range(n_values):
                dat = rbytes[i * bits_per_mask : (i + 1) * bits_per_mask]
                values[i] = values[i] - int.from_bytes(dat, "big")
        return np.array([val % self.max_int for val in values])

    def _generate_masks_numpy(
        self,
        n_values: int,
    ) -> np.ndarray:
        """Generate a number of masking values."""
        mask = np.zeros(shape=(n_values,), dtype=self._dtype)
        max_val = self.max_int
        for rng in self._pos_rng:
            mask += rng.integers(max_val, dtype=self._dtype, size=n_values)
        for rng in self._neg_rng:
            mask -= rng.integers(max_val, dtype=self._dtype, size=n_values)
        return mask

    def encrypt_uint(
        self,
        value: int,
    ) -> int:
        mask = int(self._generate_masks(1)[0])
        return (value + mask) % self.max_int

    def wrap_into_secure_aggregate(
        self,
        encrypted: List[int],
        enc_specs: EncryptedSpecs,
        cleartext: Optional[Dict[str, Any]],
        agg_cls: Type[AggregateT],
    ) -> MaskedAggregate[AggregateT]:
        return MaskedAggregate(
            encrypted=encrypted,
            enc_specs=enc_specs,
            cleartext=cleartext,
            agg_cls=agg_cls,
            max_int=self.max_int,
            n_aggrg=1,
        )
