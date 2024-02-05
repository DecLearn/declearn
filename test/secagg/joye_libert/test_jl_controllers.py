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

"""Unit tests for Joye-Libert encryption and decryption controllers."""

import dataclasses
import secrets
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest

from declearn.model.api import Vector, VectorSpec
from declearn.model.sklearn import NumpyVector
from declearn.secagg.joye_libert import (
    DEFAULT_BIPRIME,
    JLSAggregate,
    JoyeLibertDecrypter,
    JoyeLibertEncrypter,
    sum_encrypted,
)
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
    list_available_frameworks,
    to_numpy,
)
from declearn.utils import Aggregate, set_device_policy


@dataclasses.dataclass
class MockAggregate(Aggregate, base_cls=True, register=True):
    """Mock 'Aggregate' subclass for testing purposes."""

    _group_key = "mock-aggregate"

    string: str
    scalar_int: int
    scalar_float: float
    np_array: np.ndarray
    vector: Vector

    @staticmethod
    def aggregate_string(
        val_a: str,
        val_b: str,
    ) -> str:
        """Aggregation rule for the 'string' field."""
        if val_a != val_b:
            raise ValueError("Cannot aggregate mocks with distinct string.")
        return val_a

    def prepare_for_secagg(
        self,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        secagg_fields = dataclasses.asdict(self)
        clrtxt_fields = {"string": secagg_fields.pop("string")}
        return secagg_fields, clrtxt_fields


class TestJoyeLibertEncrypter:
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

    def test_encrypt_int(
        self,
    ) -> None:
        """Test that encryption of an int has proper outputs."""
        prv_key = secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
        encrypter = JoyeLibertEncrypter(prv_key)
        # Test that an integer value is encrypted into an int.
        clr_val = secrets.randbits(32)
        enc_val = encrypter.encrypt_int(clr_val)
        assert isinstance(enc_val, int) and enc_val < encrypter.biprime**2
        # Test that encrypting the same value gives a distinct output,
        # due to the increment of the internal time stamp.
        bis_val = encrypter.encrypt_int(clr_val)
        assert isinstance(bis_val, int) and bis_val < encrypter.biprime**2
        assert bis_val != enc_val

    def test_encrypt_float(
        self,
    ) -> None:
        """Test that encryption of an int has proper outputs."""
        prv_key = secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
        encrypter = JoyeLibertEncrypter(prv_key)
        # Test that a float value is encrypted into an int.
        clr_val = secrets.randbits(32) / secrets.randbits(32)
        enc_val = encrypter.encrypt_float(clr_val)
        assert isinstance(enc_val, int) and enc_val < encrypter.biprime**2
        # Test that encrypting the same value gives a distinct output,
        # due to the increment of the internal time stamp.
        bis_val = encrypter.encrypt_float(clr_val)
        assert isinstance(bis_val, int) and bis_val < encrypter.biprime**2
        assert bis_val != enc_val

    @pytest.mark.parametrize("dtype", ["int8", "int32", "float16", "float64"])
    def test_encrypt_array(
        self,
        dtype: str,
    ) -> None:
        """Test that encryption of a numpy array has proper outputs."""
        prv_key = secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
        encrypter = JoyeLibertEncrypter(prv_key)
        rng = np.random.default_rng()
        # Test that an int array is encrypted into a list of int (+ specs).
        clr_arr = rng.uniform(-10, 10, size=(8, 4)).astype(dtype)
        enc_arr, arr_spec = encrypter.encrypt_numpy_array(clr_arr)
        assert isinstance(enc_arr, list)
        assert all(
            isinstance(x, int) and (x < encrypter.biprime**2)
            for x in enc_arr
        )
        # Verify that the returned array spec matches inputs.
        assert isinstance(arr_spec, tuple) and len(arr_spec) == 2
        assert arr_spec == (list(clr_arr.shape), dtype)

    def test_encrypt_array_invalid_type(
        self,
    ) -> None:
        """Test that encryption of an object numpy array raises TypeError."""
        prv_key = secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
        encrypter = JoyeLibertEncrypter(prv_key)
        clr_val = np.array(["a", "b", "c"])
        with pytest.raises(TypeError):
            encrypter.encrypt_numpy_array(clr_val)

    @pytest.mark.parametrize("framework", list_available_frameworks())
    def test_encrypt_vector(
        self,
        framework: FrameworkType,
    ) -> None:
        """Test that encryption of a declearn Vector has proper outputs."""
        set_device_policy(gpu=False)
        prv_key = secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
        encrypter = JoyeLibertEncrypter(prv_key)
        # Test that a Vector is encrypted into a list of int (+ specs).
        clr_vec = GradientsTestCase(framework).mock_ones
        enc_vec, vec_spec = encrypter.encrypt_vector(clr_vec)
        assert isinstance(enc_vec, list)
        assert all(
            isinstance(x, int) and (x < encrypter.biprime**2)
            for x in enc_vec
        )
        # Verify that encrypted values differ, despite the use of all-1 inputs.
        assert len(set(enc_vec)) > 1
        # Verify that the returned VectorSpec matches inputs.
        assert isinstance(vec_spec, VectorSpec)
        assert vec_spec == clr_vec.get_vector_specs()

    def test_encrypt_aggregate(
        self,
    ) -> None:
        """Test that encryption of an Aggregate works properly."""
        prv_key = secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
        encrypter = JoyeLibertEncrypter(prv_key)
        # Set up a MockAggregate and encrypt it.
        rng = np.random.default_rng()
        aggregate = MockAggregate(
            string="mock",
            scalar_int=secrets.randbits(32),
            scalar_float=secrets.randbits(32) / secrets.randbits(32),
            np_array=rng.normal(size=(32, 8)),
            vector=NumpyVector(
                {"a": rng.normal(size=(16, 8)), "b": rng.normal(size=(8,))}
            ),
        )
        encrypted = encrypter.encrypt_aggregate(aggregate)
        # Test that the output has proper type and check some attributes.
        assert isinstance(encrypted, JLSAggregate)
        assert [n for n, *_ in encrypted.enc_specs] == [
            "scalar_int",
            "scalar_float",
            "np_array",
            "vector",
        ]
        assert all(isinstance(x, int) for x in encrypted.encrypted)
        assert encrypted.cleartext == {"string": aggregate.string}
        assert encrypted.agg_cls is MockAggregate
        assert encrypted.biprime == encrypter.biprime
        assert encrypted.n_aggrg == 1


@pytest.mark.parametrize("n_peers", [1, 3])
class TestJoyeLibertDecrypter:
    """Unit tests for 'declearn.secagg.joye_libert.JoyeLibertDecrypter'.

    These tests are not entirely unitary: they are designed under the
    assumption that `JoyeLibertEncrypter` works properly, and test at
    once both the formal behavior of `JoyeLibertDecrypter` and proper
    functional behavior of both controllers as a pair. I.e. while the
    unit tests for the encrypter only check that outputs abide by the
    specs in terms of type, these tests check that sum-decryption of
    encrypted values yields correct results.

    All tests are designed to run twice, once with `n_peers=1`, once
    with `n_peers=3`. Intuitively the first case verifies decryption
    in a test-only setting, while the second tackles actual SecAgg.
    """

    def test_init(
        self,
        n_peers: int,
    ) -> None:
        """Test that instantiation hyper-parameters are properly used."""
        pub_key = secrets.randbits(32)
        biprime = secrets.randbits(16)
        bitsize = 8
        clipval = 1.0
        decrypter = JoyeLibertDecrypter(
            pub_key, n_peers, biprime, bitsize, clipval
        )
        assert decrypter.pub_key == pub_key
        assert decrypter.n_peers == n_peers
        assert decrypter.biprime == biprime
        assert decrypter.quantizer.int_range == 2**bitsize - 1
        assert decrypter.quantizer.val_range == clipval

    def test_decrypt_int(
        self,
        n_peers: int,
    ) -> None:
        """Test that decryption of a sum of int works properly."""
        s_keys = [
            secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
            for _ in range(n_peers)
        ]
        decrypter = JoyeLibertDecrypter(pub_key=-sum(s_keys), n_peers=n_peers)
        # Encrypt and aggregate random int values.
        cleartext = [secrets.randbits(32) for _ in range(n_peers)]
        encrypted = [
            JoyeLibertEncrypter(key).encrypt_int(val)
            for key, val in zip(s_keys, cleartext)
        ]
        # Test that decryption works properly.
        decrypted = decrypter.decrypt_int(sum_encrypted(encrypted))
        assert isinstance(decrypted, int)
        assert decrypted == sum(cleartext)

    def test_decrypt_float(
        self,
        n_peers: int,
    ) -> None:
        """Test that decryption of a sum of float works properly."""
        s_keys = [
            secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
            for _ in range(n_peers)
        ]
        decrypter = JoyeLibertDecrypter(pub_key=-sum(s_keys), n_peers=n_peers)
        # Encrypt and aggregate random float values.
        cleartext = [
            secrets.randbits(32) / secrets.randbits(32) for _ in range(n_peers)
        ]
        encrypted = [
            JoyeLibertEncrypter(key).encrypt_float(val)
            for key, val in zip(s_keys, cleartext)
        ]
        # Test that decryption works properly.
        decrypted = decrypter.decrypt_float(sum_encrypted(encrypted))
        assert isinstance(decrypted, float)
        assert abs(decrypted - sum(cleartext)) < 1e-10

    @pytest.mark.parametrize("dtype", ["uint8", "int64", "float16", "float64"])
    def test_decrypt_array(
        self,
        dtype: str,
        n_peers: int,
    ) -> None:
        """Test that decryption of a sum of numpy array works properly."""
        s_keys = [
            secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
            for _ in range(n_peers)
        ]
        decrypter = JoyeLibertDecrypter(pub_key=-sum(s_keys), n_peers=n_peers)
        rng = np.random.default_rng()
        # Encrypt and aggregate random numpy arrays.
        low = 0 if dtype.startswith("u") else -10
        cleartext = [
            rng.uniform(low, 10, size=(8, 4)).astype(dtype)
            for _ in range(n_peers)
        ]
        encrypted = [
            JoyeLibertEncrypter(key).encrypt_numpy_array(val)
            for key, val in zip(s_keys, cleartext)
        ]
        sum_values = [
            sum_encrypted(val) for val in zip(*(val for val, _ in encrypted))
        ]
        # Test that decryption works properly.
        decrypted = decrypter.decrypt_numpy_array(
            values=sum_values, specs=encrypted[0][1]
        )
        assert isinstance(decrypted, np.ndarray)
        assert decrypted.shape == cleartext[0].shape
        assert decrypted.dtype == dtype
        if dtype == "float16":
            assert np.allclose(decrypted, sum(cleartext), atol=0.05)
        else:
            assert np.allclose(decrypted, sum(cleartext))

    @pytest.mark.parametrize("framework", list_available_frameworks())
    def test_decrypt_vector(
        self,
        framework: FrameworkType,
        n_peers: int,
    ) -> None:
        """Test that decryption of a sum of declearn Vector works properly."""
        set_device_policy(gpu=False)
        s_keys = [
            secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
            for _ in range(n_peers)
        ]
        decrypter = JoyeLibertDecrypter(pub_key=-sum(s_keys), n_peers=n_peers)
        test_case = GradientsTestCase(framework)
        # Encrypt and aggregate Vector objects.
        cleartext = [test_case.mock_ones for _ in range(n_peers)]
        encrypted = [
            JoyeLibertEncrypter(key).encrypt_vector(val)
            for key, val in zip(s_keys, cleartext)
        ]
        sum_values = [
            sum_encrypted(val) for val in zip(*(val for val, _ in encrypted))
        ]
        # Test that decryption works properly.
        decrypted = decrypter.decrypt_vector(
            values=sum_values, specs=encrypted[0][1]
        )
        assert isinstance(decrypted, test_case.vector_cls)
        clear_sum = sum(cleartext[1:], start=cleartext[0])
        assert decrypted.get_vector_specs() == clear_sum.get_vector_specs()
        assert all(
            np.allclose(
                to_numpy(val, framework),
                to_numpy(decrypted.coefs[key], framework),
            )
            for key, val in clear_sum.coefs.items()
        )

    def test_decrypt_aggregate(
        self,
        n_peers: int,
    ) -> None:
        """Test that decryption of a sum of Aggregates works properly."""
        s_keys = [
            secrets.randbits(2 * DEFAULT_BIPRIME.bit_length())
            for _ in range(n_peers)
        ]
        decrypter = JoyeLibertDecrypter(pub_key=-sum(s_keys), n_peers=n_peers)
        # Encrypt and aggregate MockAggregate objects.
        rng = np.random.default_rng()
        cleartext = [
            MockAggregate(
                string="mock",
                scalar_int=secrets.randbits(32),
                scalar_float=secrets.randbits(32) / secrets.randbits(32),
                np_array=rng.normal(size=(32, 8)),
                vector=NumpyVector(
                    {"a": rng.normal(size=(16, 8)), "b": rng.normal(size=(8,))}
                ),
            )
            for _ in range(n_peers)
        ]
        encrypted = [
            JoyeLibertEncrypter(key).encrypt_aggregate(val)
            for key, val in zip(s_keys, cleartext)
        ]
        sum_aggrg = sum(encrypted[1:], start=encrypted[0])
        # Test that decryption works properly.
        decrypted = decrypter.decrypt_aggregate(sum_aggrg)
        assert isinstance(decrypted, MockAggregate)
        aggregate = sum(cleartext[1:], start=cleartext[0])
        assert decrypted.string == aggregate.string
        assert decrypted.scalar_int == aggregate.scalar_int
        assert abs(decrypted.scalar_float - aggregate.scalar_float) < 1e-10
        assert np.allclose(decrypted.np_array, aggregate.np_array)
        assert decrypted.np_array.dtype == aggregate.np_array.dtype
        assert all(
            np.allclose(val, aggregate.vector.coefs[key])
            for key, val in decrypted.vector.coefs.items()
        )
