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

"""Unit tests for Masking-based SecAgg controllers."""

import copy
import os
import secrets
from typing import Any, Dict, List, Tuple

import pytest

from declearn.secagg.masking import (
    MaskedAggregate,
    MaskingDecrypter,
    MaskingEncrypter,
)
from declearn.test_utils import make_importable

with make_importable(os.path.join(os.path.dirname(__file__))):
    from secagg_testing import (
        DecrypterExceptionsTestSuite,
        DecrypterTestSuite,
        EncrypterTestSuite,
        MockSimpleAggregate,
        SecureAggregateTestSuite,
    )


class TestMaskingEncrypter(EncrypterTestSuite[MaskingEncrypter]):
    """Unit tests for 'declearn.secagg.masking.MaskingEncrypter'."""

    def setup_encrypter(
        self,
        bitsize: int = 32,
    ) -> Tuple[MaskingEncrypter, int]:
        pos_masks_seeds = [0]
        neg_masks_seeds = [1]
        encrypter = MaskingEncrypter(
            pos_masks_seeds, neg_masks_seeds, bitsize=bitsize
        )
        max_value = encrypter.max_int
        return encrypter, max_value

    @pytest.mark.parametrize(
        "large_quantizer_field", [True, False], ids=["quant128", "quant64"]
    )
    @pytest.mark.parametrize("dtype", ["int8", "int32", "float16", "float64"])
    def test_encrypt_array(
        self,
        dtype: str,
        large_quantizer_field: bool,
    ) -> None:
        if large_quantizer_field:
            with pytest.raises(ValueError):
                self.setup_encrypter(bitsize=128)
        else:
            super().test_encrypt_array(dtype, large_quantizer_field)


@pytest.mark.parametrize("n_peers", [2, 5])
class TestMaskingDecrypter(
    DecrypterTestSuite[MaskingDecrypter, MaskingEncrypter]
):
    """Functional tests for Masking SecAgg controllers."""

    def sum_encrypted(
        self,
        encrypted: List[int],
    ) -> int:
        return sum(encrypted)

    def setup_decrypter_and_encrypters(
        self,
        n_peers: int,
    ) -> Tuple[MaskingDecrypter, List[MaskingEncrypter]]:
        decrypter = MaskingDecrypter(n_peers=n_peers)
        rng_seeds = [
            ([], []) for _ in range(n_peers)
        ]  # type: List[Tuple[List[int], List[int]]]
        counter = 0
        for i in range(n_peers - 1):
            for j in range(i + 1, n_peers):
                rng_seeds[i][0].append(counter)
                rng_seeds[j][1].append(counter)
                counter += 1
        encrypters = [
            MaskingEncrypter(pos_masks_seeds, neg_masks_seeds)
            for pos_masks_seeds, neg_masks_seeds in rng_seeds
        ]
        return decrypter, encrypters


class TestMaskingDecrypterExceptions(
    DecrypterExceptionsTestSuite[MaskingDecrypter]
):
    """Unit tests for exception-raising 'MaskingDecrypter' uses."""

    def setup_decrypter(
        self,
    ) -> Tuple[MaskingDecrypter, int, Dict[str, Any]]:
        decrypter = MaskingDecrypter(n_peers=2)
        max_value = decrypter.max_int
        kwargs = {"max_int": decrypter.max_int}
        return decrypter, max_value, kwargs

    def test_decrypt_aggregate_error_invalid_max_int(
        self,
    ) -> None:
        """Test that decryption with mismatching int field raises properly."""
        decrypter, max_value, _ = self.setup_decrypter()
        encrypted = MaskedAggregate(
            encrypted=[secrets.randbelow(max_value)],
            enc_specs=[("value", 1, True)],
            cleartext=None,
            agg_cls=MockSimpleAggregate,
            max_int=decrypter.max_int - 1,
            n_aggrg=decrypter.n_peers,
        )
        with pytest.raises(ValueError):
            decrypter.decrypt_aggregate(encrypted)


class TestMaskedAggregate(SecureAggregateTestSuite[MaskedAggregate]):
    """Unit tests on the 'MaskedAggregate' data structure."""

    def setup_secure_aggregate(
        self,
    ) -> MaskedAggregate:
        return MaskedAggregate(
            encrypted=[secrets.randbelow(2**64)],
            enc_specs=[("value", 1, False)],
            cleartext=None,
            agg_cls=MockSimpleAggregate,
            max_int=2**64,
            n_aggrg=1,
        )

    def test_aggregate_error_invalid_max_int(
        self,
    ) -> None:
        """Test that MaskedAggregate aggregation raises on distinct max int."""
        msk_agg = self.setup_secure_aggregate()
        msk_bis = copy.deepcopy(msk_agg)
        msk_bis.max_int += 1
        with pytest.raises(ValueError):
            msk_agg.aggregate(msk_bis)
