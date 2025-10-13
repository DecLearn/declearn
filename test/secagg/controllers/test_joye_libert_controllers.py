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

"""Unit tests for Joye-Libert SecAgg controllers."""

import copy
import os
import secrets
from typing import Any, Dict, List, Tuple

import pytest

from declearn.secagg.joye_libert import (
    DEFAULT_BIPRIME,
    JLSAggregate,
    JoyeLibertDecrypter,
    JoyeLibertEncrypter,
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


class TestJoyeLibertEncrypter(EncrypterTestSuite):
    """Unit tests for 'declearn.secagg.joye_libert.JoyeLibertEncrypter'."""

    def test_init(
        self,
    ) -> None:
        """Test that instantiation hyper-parameters are properly used."""
        prv_key = secrets.randbits(32)
        biprime = secrets.randbits(16)
        bitsize = 8
        clipval = 1.0
        encrypter = JoyeLibertEncrypter(prv_key, biprime, bitsize, clipval)
        assert encrypter.prv_key == prv_key
        assert encrypter.biprime == biprime
        assert encrypter.quantizer.int_range == 2**bitsize - 1
        assert encrypter.quantizer.val_range == clipval

    def setup_encrypter(
        self,
        bitsize: int = 32,
    ) -> Tuple[JoyeLibertEncrypter, int]:
        prv_key = secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
        encrypter = JoyeLibertEncrypter(prv_key, bitsize=bitsize)
        max_value = encrypter.biprime**2
        return encrypter, max_value


@pytest.mark.parametrize("n_peers", [1, 3])
class TestJoyeLibertDecrypter(DecrypterTestSuite):
    """Functional tests for Joye-Libert SecAgg controllers."""

    def setup_decrypter_and_encrypters(
        self,
        n_peers: int,
    ) -> Tuple[JoyeLibertDecrypter, List[JoyeLibertEncrypter]]:
        s_keys = [
            secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
            for _ in range(n_peers)
        ]
        decrypter = JoyeLibertDecrypter(pub_key=-sum(s_keys), n_peers=n_peers)
        encrypters = [JoyeLibertEncrypter(prv_key) for prv_key in s_keys]
        return decrypter, encrypters


class TestJoyeLibertDecrypterExceptions(DecrypterExceptionsTestSuite):
    """Unit tests for exception-raising 'JoyeLibertDecrypter' uses."""

    def setup_decrypter(
        self,
    ) -> Tuple[JoyeLibertDecrypter, int, Dict[str, Any]]:
        pub_key = -secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
        decrypter = JoyeLibertDecrypter(pub_key=pub_key, n_peers=1)
        max_value = decrypter.biprime**2
        kwargs = {"biprime": decrypter.biprime}
        return decrypter, max_value, kwargs

    def test_decrypt_aggregate_error_invalid_biprime(
        self,
    ) -> None:
        """Test that decryption with mismatching biprime raises properly."""
        decrypter, max_value, _ = self.setup_decrypter()
        encrypted = JLSAggregate(
            encrypted=[secrets.randbelow(max_value)],
            enc_specs=[("value", 1, True)],
            cleartext=None,
            agg_cls=MockSimpleAggregate,
            biprime=decrypter.biprime - 1,  # invalid biprime here
            n_aggrg=decrypter.n_peers,
        )
        with pytest.raises(ValueError):
            decrypter.decrypt_aggregate(encrypted)


class TestJLSAggregate(SecureAggregateTestSuite):
    """Unit tests on the 'JLSAggregate' data structure."""

    def setup_secure_aggregate(
        self,
    ) -> JLSAggregate:
        return JLSAggregate(
            encrypted=[secrets.randbelow(DEFAULT_BIPRIME**2)],
            enc_specs=[("value", 1, False)],
            cleartext=None,
            agg_cls=MockSimpleAggregate,
            biprime=DEFAULT_BIPRIME,
            n_aggrg=1,
        )

    def test_aggregate_error_invalid_biprime(
        self,
    ) -> None:
        """Test that JLSAggregate aggregation raises on distinct biprime."""
        jls_agg = self.setup_secure_aggregate()
        jls_bis = copy.deepcopy(jls_agg)
        jls_bis.biprime += 1
        with pytest.raises(ValueError):
            jls_agg.aggregate(jls_bis)
