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

    TODO: Add references and algorithm details.
    """

    def __init__(
        self,
        pos_masks_seeds: List[int],
        neg_masks_seeds: List[int],
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        super().__init__(bitsize=bitsize, clipval=clipval)
        if not self.quantizer.numpy_compatible:
            raise ValueError(
                "'MaskingEncrypter' requires 'bitsize' to be low enough for "
                "compatibility with numpy uint dtypes. This usually means a "
                "bitsize <= 64."
            )
        self.max_int = 2**bitsize
        self._dtype = get_numpy_uint_dtype(self.quantizer.int_range)
        self._pos_rng = [
            np.random.default_rng(seed) for seed in pos_masks_seeds
        ]
        self._neg_rng = [
            np.random.default_rng(seed) for seed in neg_masks_seeds
        ]

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

    def encrypt_uint(
        self,
        value: int,
    ) -> int:
        mask = int(self._generate_masks(1)[0])
        return value + mask

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
