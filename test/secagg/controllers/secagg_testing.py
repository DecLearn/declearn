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

"""Unit tests template for encryption and decryption controllers."""

import abc
import copy
import dataclasses
import os
import secrets
from typing import Any, Collection, Dict, Optional, Tuple, Union
from unittest import mock

import numpy as np
import pytest

from declearn.model.api import Vector, VectorSpec
from declearn.model.sklearn import NumpyVector
from declearn.secagg.api import Decrypter, Encrypter, SecureAggregate
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
    assert_json_serializable_dict,
    list_available_frameworks,
    to_numpy,
)
from declearn.utils import Aggregate, json_dump, json_load, set_device_policy


@dataclasses.dataclass
class MockAggregate(
    Aggregate,
    base_cls=True,  # type: ignore[call-arg]  # false-positive
    register=True,  # type: ignore[call-arg]  # false-positive
):
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


class EncrypterTestSuite(metaclass=abc.ABCMeta):
    """Unit tests for 'declearn.secagg.api.Encrypter' subclasses."""

    @abc.abstractmethod
    def setup_encrypter(
        self,
        bitsize: int = 32,
    ) -> Tuple[Encrypter, int]:
        """Set up an Encrypter.

        Returns
        -------
        encrypter:
            `Encrypter` instance, parametrized with `bitsize`.
        max_value:
            Maximal expected value for encrypted integers.
        """

    def test_encrypt_uint(
        self,
    ) -> None:
        """Test that encryption of an int has proper outputs."""
        encrypter, max_value = self.setup_encrypter()
        # Test that an integer value is encrypted into an int.
        clr_val = secrets.randbits(32)
        enc_val = encrypter.encrypt_uint(clr_val)
        assert isinstance(enc_val, int) and enc_val < max_value
        # Test that encrypting the same value gives a distinct output,
        # due to the increment of the internal time stamp.
        bis_val = encrypter.encrypt_uint(clr_val)
        assert isinstance(bis_val, int) and bis_val < max_value
        assert bis_val != enc_val

    def test_encrypt_float(
        self,
    ) -> None:
        """Test that encryption of an int has proper outputs."""
        encrypter, max_value = self.setup_encrypter()
        # Test that a float value is encrypted into an int.
        clr_val = secrets.randbits(32) / secrets.randbits(32)
        enc_val = encrypter.encrypt_float(clr_val)
        assert isinstance(enc_val, int) and enc_val < max_value
        # Test that encrypting the same value gives a distinct output,
        # due to the increment of the internal time stamp.
        bis_val = encrypter.encrypt_float(clr_val)
        assert isinstance(bis_val, int) and bis_val < max_value
        assert bis_val != enc_val

    @pytest.mark.parametrize(
        "large_quantizer_field", [True, False], ids=["quant128", "quant64"]
    )
    @pytest.mark.parametrize("dtype", ["int8", "int32", "float16", "float64"])
    def test_encrypt_array(
        self,
        dtype: str,
        large_quantizer_field: bool,
    ) -> None:
        """Test that encryption of a numpy array has proper outputs."""
        encrypter, max_value = self.setup_encrypter(
            bitsize=128 if large_quantizer_field else 64
        )
        rng = np.random.default_rng()
        # Test that an int array is encrypted into a list of int (+ specs).
        clr_arr = rng.uniform(-10, 10, size=(8, 4)).astype(dtype)
        enc_arr, arr_spec = encrypter.encrypt_numpy_array(clr_arr)
        assert isinstance(enc_arr, list)
        assert all(isinstance(x, int) and (x < max_value) for x in enc_arr)
        # Verify that the returned array spec matches inputs.
        assert isinstance(arr_spec, tuple) and len(arr_spec) == 2
        assert arr_spec == (list(clr_arr.shape), dtype)

    def test_encrypt_array_invalid_type(
        self,
    ) -> None:
        """Test that encryption of an object numpy array raises TypeError."""
        encrypter, _ = self.setup_encrypter()
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
        encrypter, max_value = self.setup_encrypter()
        # Test that a Vector is encrypted into a list of int (+ specs).
        clr_vec = GradientsTestCase(framework).mock_ones
        enc_vec, vec_spec = encrypter.encrypt_vector(clr_vec)
        assert isinstance(enc_vec, list)
        assert all(isinstance(x, int) and (x < max_value) for x in enc_vec)
        # Verify that encrypted values differ, despite the use of all-1 inputs.
        assert len(set(enc_vec)) > 1
        # Verify that the returned VectorSpec matches inputs.
        assert isinstance(vec_spec, VectorSpec)
        assert vec_spec == clr_vec.get_vector_specs()

    def test_encrypt_aggregate(
        self,
    ) -> None:
        """Test that encryption of an Aggregate works properly."""
        encrypter, _ = self.setup_encrypter()
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
        assert isinstance(encrypted, SecureAggregate)
        assert [n for n, *_ in encrypted.enc_specs] == [
            "scalar_int",
            "scalar_float",
            "np_array",
            "vector",
        ]
        assert all(isinstance(x, int) for x in encrypted.encrypted)
        assert encrypted.cleartext == {"string": aggregate.string}
        assert encrypted.agg_cls is MockAggregate
        assert encrypted.n_aggrg == 1

    def test_encrypt_aggregate_with_invalid_type(
        self,
    ) -> None:
        """Test error raising when trying to encrypt unsupported types."""
        encrypter, _ = self.setup_encrypter()
        # Set up a MockAggregate with a non-Vector MagicMock 'vector' field.
        aggregate = MockAggregate(
            string="mock",
            scalar_int=0,
            scalar_float=0.0,
            np_array=np.array([0.0]),
            vector=mock.MagicMock(),
        )
        # Test that the latter field raises a TypeError.
        with pytest.raises(TypeError):
            encrypter.encrypt_aggregate(aggregate)


class DecrypterTestSuite(metaclass=abc.ABCMeta):
    """Unit tests for 'declearn.secagg.api.Decrypter' subclasses.

    These tests are not entirely unitary: they are designed under the
    assumption that the related `Encrypter` works properly, and test
    both the formal behavior of the `Decrypter` and proper functional
    behavior of both controllers as a pair.

    I.e. while the unit tests for the encrypter only check that outputs
    abide by the specs in terms of type and maximum value, these tests
    check that sum-decryption of encrypted values yields correct results.

    All tests are designed to run twice, once with `n_peers=1`, once
    with `n_peers=3`. Intuitively the first case verifies decryption
    in a test-only setting, while the second tackles actual SecAgg.
    """

    @abc.abstractmethod
    def setup_decrypter_and_encrypters(
        self,
        n_peers: int,
    ) -> Tuple[Decrypter, Collection[Encrypter]]:
        """Set up a Decrypter and an ensemble of Encrypters."""

    def test_decrypt_uint(
        self,
        n_peers: int,
    ) -> None:
        """Test that decryption of a sum of int works properly."""
        decrypter, encrypters = self.setup_decrypter_and_encrypters(n_peers)
        # Encrypt and aggregate random int values.
        cleartext = [secrets.randbits(32) for _ in range(n_peers)]
        encrypted = [
            encrypter.encrypt_uint(value)
            for encrypter, value in zip(encrypters, cleartext, strict=False)
        ]
        # Test that decryption works properly.
        decrypted = decrypter.decrypt_uint(decrypter.sum_encrypted(encrypted))
        assert isinstance(decrypted, int)
        assert decrypted == sum(cleartext)

    def test_decrypt_float(
        self,
        n_peers: int,
    ) -> None:
        """Test that decryption of a sum of float works properly."""
        decrypter, encrypters = self.setup_decrypter_and_encrypters(n_peers)
        # Encrypt and aggregate random float values.
        cleartext = [
            secrets.randbits(32) / secrets.randbits(32) for _ in range(n_peers)
        ]
        encrypted = [
            encrypter.encrypt_float(value)
            for encrypter, value in zip(encrypters, cleartext, strict=False)
        ]
        # Test that decryption works properly.
        decrypted = decrypter.decrypt_float(decrypter.sum_encrypted(encrypted))
        assert isinstance(decrypted, float)
        assert abs(decrypted - sum(cleartext)) < 1e-10

    @pytest.mark.parametrize("dtype", ["uint8", "int64", "float16", "float64"])
    def test_decrypt_array(
        self,
        dtype: str,
        n_peers: int,
    ) -> None:
        """Test that decryption of a sum of numpy array works properly."""
        decrypter, encrypters = self.setup_decrypter_and_encrypters(n_peers)
        rng = np.random.default_rng()
        # Encrypt and aggregate random numpy arrays.
        low = 0 if dtype.startswith("u") else -10
        cleartext = [
            rng.uniform(low, 10, size=(8, 4)).astype(dtype)
            for _ in range(n_peers)
        ]
        encrypted = [
            encrypter.encrypt_numpy_array(value)
            for encrypter, value in zip(encrypters, cleartext, strict=False)
        ]
        sum_values = [
            decrypter.sum_encrypted(values)  # type: ignore  # false-positive
            for values in zip(*(val for val, _ in encrypted), strict=False)
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
        decrypter, encrypters = self.setup_decrypter_and_encrypters(n_peers)
        test_case = GradientsTestCase(framework)
        # Encrypt and aggregate Vector objects.
        cleartext = [test_case.mock_ones for _ in range(n_peers)]
        encrypted = [
            encrypter.encrypt_vector(value)
            for encrypter, value in zip(encrypters, cleartext, strict=False)
        ]
        sum_values = [
            decrypter.sum_encrypted(values)  # type: ignore  # false-positive
            for values in zip(*(val for val, _ in encrypted), strict=False)
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
        decrypter, encrypters = self.setup_decrypter_and_encrypters(n_peers)
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
            encrypter.encrypt_aggregate(value)
            for encrypter, value in zip(encrypters, cleartext, strict=False)
        ]
        sum_aggrg = sum(encrypted[1:], start=encrypted[0])
        # Test that decryption works properly.
        decrypted = decrypter.decrypt_aggregate(sum_aggrg)
        assert isinstance(decrypted, MockAggregate)
        aggregate = sum(cleartext[1:], start=cleartext[0])
        assert isinstance(aggregate, MockAggregate)  # prevent mypy false-pos.
        assert decrypted.string == aggregate.string
        assert decrypted.scalar_int == aggregate.scalar_int
        assert abs(decrypted.scalar_float - aggregate.scalar_float) < 1e-10
        assert np.allclose(decrypted.np_array, aggregate.np_array)
        assert decrypted.np_array.dtype == aggregate.np_array.dtype
        assert all(
            np.allclose(val, aggregate.vector.coefs[key])
            for key, val in decrypted.vector.coefs.items()
        )


@dataclasses.dataclass
class MockSimpleAggregate(
    Aggregate,
    base_cls=True,  # type: ignore[call-arg]  # false-positive
    register=True,  # type: ignore[call-arg]  # false-positive
):
    """Simple mock Aggregate child class."""

    _group_key = "mock-simple-aggregate"

    value: Union[int, float]


class DecrypterExceptionsTestSuite(metaclass=abc.ABCMeta):
    """Unit tests for exception-raising 'JoyeLibertDecrypter' uses."""

    @abc.abstractmethod
    def setup_decrypter(
        self,
    ) -> Tuple[Decrypter, int, Dict[str, Any]]:
        """Set up a Decrypter instance and some metatdata.

        Returns
        -------
        decrypter:
            `Decrypter` instance.
        max_value:
            Maximal value for encrypted integers.
        kwargs:
            Dict of algorithm-specific kwargs to set up a valid
            `SecureAggregate` input for the returned decrypter.
        """

    def test_decrypt_aggregate_error_invalid_type(
        self,
    ) -> None:
        """Test that decryption of a non- JLSAggregate raises properly."""
        decrypter, *_ = self.setup_decrypter()
        aggregate = mock.MagicMock()
        with pytest.raises(TypeError):
            decrypter.decrypt_aggregate(aggregate)

    def test_mock_aggregate_validity(
        self,
    ) -> None:
        """Merely test that the Aggregate sublass used can work properly."""
        decrypter, max_value, kwargs = self.setup_decrypter()
        encrypted = decrypter.secure_aggregate_cls(
            encrypted=[secrets.randbelow(max_value)],
            enc_specs=[("value", 1, True)],
            cleartext=None,
            agg_cls=MockSimpleAggregate,
            n_aggrg=decrypter.n_peers,
            **kwargs,
        )
        decrypted = decrypter.decrypt_aggregate(encrypted)
        assert isinstance(decrypted, MockSimpleAggregate)
        assert isinstance(decrypted.value, float)

    def test_decrypt_aggregate_error_invalid_n_aggrg(
        self,
    ) -> None:
        """Test that decryption of a non- JLSAggregate raises properly."""
        decrypter, max_value, kwargs = self.setup_decrypter()
        encrypted = decrypter.secure_aggregate_cls(
            encrypted=[secrets.randbelow(max_value)],
            enc_specs=[("value", 1, True)],
            cleartext=None,
            agg_cls=MockSimpleAggregate,
            n_aggrg=decrypter.n_peers + 1,  # invalid n_aggrg here
            **kwargs,
        )
        with pytest.raises(ValueError):
            decrypter.decrypt_aggregate(encrypted)

    def test_decrypt_aggregate_error_invalid_field_type(
        self,
    ) -> None:
        """Test that decryption of a non- JLSAggregate raises properly."""
        decrypter, max_value, kwargs = self.setup_decrypter()
        encrypted = decrypter.secure_aggregate_cls(
            encrypted=[secrets.randbelow(max_value)],
            enc_specs=[("value", 1, mock.MagicMock())],  # invalid specs here
            cleartext=None,
            agg_cls=MockSimpleAggregate,
            n_aggrg=decrypter.n_peers,
            **kwargs,
        )
        with pytest.raises(TypeError):
            decrypter.decrypt_aggregate(encrypted)


class SecureAggregateTestSuite(metaclass=abc.ABCMeta):
    """Unit tests for 'declearn.secagg.api.SecureAggregate' subclasses."""

    @abc.abstractmethod
    def setup_secure_aggregate(
        self,
    ) -> SecureAggregate:
        """Setup a SecureAggregate wrapping a 'MockSimpleAggregate'."""

    def test_dict_serialization(
        self,
    ) -> None:
        """Test that dict-serialization of a SecureAggregate works properly."""
        sec_agg = self.setup_secure_aggregate()
        sec_dict = sec_agg.to_dict()
        assert_json_serializable_dict(sec_dict)
        agg_bis = type(sec_agg).from_dict(sec_dict)
        assert isinstance(agg_bis, type(sec_agg))
        assert agg_bis.to_dict() == sec_dict

    def test_json_serialization(
        self,
        tmp_path: str,
    ) -> None:
        """Test that JSON-serialization of a SecureAggregate works properly."""
        sec_agg = self.setup_secure_aggregate()
        path = os.path.join(tmp_path, "agg.json")
        json_dump(sec_agg, path)
        agg_bis = json_load(path)
        assert isinstance(agg_bis, type(sec_agg))
        assert agg_bis.to_dict() == sec_agg.to_dict()

    def test_dict_deserialization_error(
        self,
    ) -> None:
        """Test that dict-deserialization exceptions are caught."""
        # Test that the KeyError due to missing data is wrapped as TypeError.
        sec_agg = self.setup_secure_aggregate()
        sec_dict = sec_agg.to_dict()
        sec_dict.pop("encrypted")
        with pytest.raises(TypeError):
            type(sec_agg).from_dict(sec_dict)

    def test_aggregate(
        self,
    ) -> None:
        """Test that SecureAggregate aggregation (summation) works properly.

        We already have functional tests, hence this test is quite limited.
        """
        sec_agg = self.setup_secure_aggregate()
        result = sec_agg + sec_agg
        assert isinstance(result, type(sec_agg))
        assert result.n_aggrg == 2

    def test_aggregate_error_invalid_type(
        self,
    ) -> None:
        """Test that SecureAggregate aggregation raises on improper types."""
        sec_agg = self.setup_secure_aggregate()
        mock_agg = mock.MagicMock()
        mock_agg.__radd__.side_effect = NotImplementedError
        # Test that type is properly checked in `aggregate`.
        with pytest.raises(TypeError):
            sec_agg.aggregate(mock_agg)
        # Test that type is properly checked in `__add__`.
        with pytest.raises(NotImplementedError):
            sec_agg + mock_agg  # pylint: disable=pointless-statement

    def test_aggregate_error_invalid_specs(
        self,
    ) -> None:
        """Test that SecureAggregate aggregation raises on distinct specs."""
        sec_agg = self.setup_secure_aggregate()
        agg_bis = copy.deepcopy(sec_agg)
        agg_bis.enc_specs = [("value", 1, True)]
        with pytest.raises(ValueError):
            sec_agg.aggregate(agg_bis)
