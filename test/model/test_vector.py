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

"""Unit tests for Vector and its subclasses.

This test makes use of `declearn.test_utils.list_available_frameworks`
so as to modularly run a standard test suite on the available Vector
subclasses.
"""

import json

import numpy as np
import pytest

from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
    list_available_frameworks,
)
from declearn.utils import json_pack, json_unpack, set_device_policy


set_device_policy(gpu=False)  # run Vector unit tests on CPU only


@pytest.fixture(name="framework", params=list_available_frameworks())
def framework_fixture(request):
    """Fixture to provide with the name of a model framework."""
    return request.param


class TestVectorAbstractMethods:
    """Test abstract methods."""

    def test_sum(self, framework: FrameworkType) -> None:
        """Test coefficient-wise sum."""
        grad = GradientsTestCase(framework)
        ones = grad.mock_ones
        test_coefs = ones.sum().coefs
        test_values = [grad.to_numpy(test_coefs[el]) for el in test_coefs]
        values = [25.0, 4.0, 1.0]
        assert values == test_values

    def test_max(self, framework: FrameworkType) -> None:
        """Test coef.-wise, element-wise maximum wrt to another Vector."""
        grad = GradientsTestCase(framework)
        ones, zeros = (grad.mock_ones, grad.mock_zeros)
        values = [np.ones((5, 5)), np.ones((4,)), np.ones((1,))]
        # test Vector
        test_coefs = zeros.maximum(ones).coefs
        test_values = [grad.to_numpy(test_coefs[el]) for el in test_coefs]
        assert all(
            (values[i] == test_values[i]).all() for i in range(len(values))
        )
        # test float
        test_coefs = zeros.maximum(1.0).coefs
        test_values = [grad.to_numpy(test_coefs[el]) for el in test_coefs]
        assert all(
            (values[i] == test_values[i]).all() for i in range(len(values))
        )

    def test_min(self, framework: FrameworkType) -> None:
        """Test coef.-wise, element-wise minimum wrt to another Vector."""
        grad = GradientsTestCase(framework)
        ones, zeros = (grad.mock_ones, grad.mock_zeros)
        values = [np.zeros((5, 5)), np.zeros((4,)), np.zeros((1,))]
        # test Vector
        test_coefs = ones.minimum(zeros).coefs
        test_values = [grad.to_numpy(test_coefs[el]) for el in test_coefs]
        assert all(
            (values[i] == test_values[i]).all() for i in range(len(values))
        )
        # test float
        test_coefs = zeros.minimum(1.0).coefs
        test_values = [grad.to_numpy(test_coefs[el]) for el in test_coefs]
        assert all(
            (values[i] == test_values[i]).all() for i in range(len(values))
        )

    def test_sign(self, framework: FrameworkType) -> None:
        """Test coefficient-wise sign check"""
        grad = GradientsTestCase(framework)
        ones = grad.mock_ones
        for vec in ones, -1 * ones:
            test_coefs = vec.sign().coefs
            test_values = [grad.to_numpy(test_coefs[el]) for el in test_coefs]
            values = [grad.to_numpy(vec.coefs[el]) for el in vec.coefs]
            assert all(
                (values[i] == test_values[i]).all() for i in range(len(values))
            )

    def test_eq(self, framework: FrameworkType) -> None:
        """Test __eq__ operator"""
        grad = GradientsTestCase(framework)
        ones, ones_bis, zeros = grad.mock_ones, grad.mock_ones, grad.mock_zeros
        rand = grad.mock_gradient
        assert ones == ones_bis
        assert zeros != ones
        assert ones != rand
        assert 1.0 != ones


class TestVector:
    """Test non-abstract methods"""

    def test_operator(self, framework: FrameworkType) -> None:
        "Test all element-wise operators wiring"
        grad = GradientsTestCase(framework)

        def _get_sq_root_two(ones, zeros):
            """Returns the comaprison of a hardcoded sequence of operations
            with its exptected result"""
            values = [
                el * (2 ** (1 / 2))
                for el in [np.ones((5, 5)), np.ones((4,)), np.ones((1,))]
            ]
            test_grad = (0 + (1.0 * ones + ones * 1.0) * ones / 1.0 - 0) ** (
                (zeros + ones - zeros) / (ones + ones)
            )
            test_coefs = test_grad.coefs
            test_values = [grad.to_numpy(test_coefs[el]) for el in test_coefs]
            return all(
                (values[i] == test_values[i]).all() for i in range(len(values))
            )

        ones, zeros = grad.mock_ones, grad.mock_zeros
        assert _get_sq_root_two(ones, zeros)
        assert _get_sq_root_two(ones, 0)

    def test_pack(self, framework: FrameworkType) -> None:
        """Test that `Vector.pack` returns JSON-serializable results."""
        grad = GradientsTestCase(framework)
        ones = grad.mock_ones
        packed = ones.pack()
        # Check that the output is a dict with str keys.
        assert isinstance(packed, dict)
        assert all(isinstance(key, str) for key in packed)
        # Check that the "packed" dict is JSON-serializable.
        dump = json.dumps(packed, default=json_pack)
        load = json.loads(dump, object_hook=json_unpack)
        assert isinstance(load, dict)
        assert load.keys() == packed.keys()
        assert all(np.all(load[key] == packed[key]) for key in load)

    def test_unpack(self, framework: FrameworkType) -> None:
        """Test that `Vector.unpack` counterparts `Vector.pack` adequately."""
        grad = GradientsTestCase(framework)
        ones = grad.mock_ones
        packed = ones.pack()
        test_vec = grad.vector_cls.unpack(packed)
        assert test_vec == ones

    def test_repr(self, framework: FrameworkType) -> None:
        """Test shape and dtypes together using __repr__"""
        grad = GradientsTestCase(framework)
        test_value = repr(grad.mock_ones)
        value = grad.mock_ones.coefs["0"]
        arr_type = f"{type(value).__module__}.{type(value).__name__}"
        value = (
            f"{grad.vector_cls.__name__} with 3 coefs:"
            f"\n    0: float64 {arr_type} with shape (5, 5)"
            f"\n    1: float64 {arr_type} with shape (4,)"
            f"\n    2: float64 {arr_type} with shape (1,)"
        )
        assert test_value == value

    def test_json_serialization(self, framework: FrameworkType) -> None:
        """Test that a Vector instance is JSON-serializable."""
        vector = GradientsTestCase(framework).mock_gradient
        dump = json.dumps(vector, default=json_pack)
        loaded = json.loads(dump, object_hook=json_unpack)
        assert isinstance(loaded, type(vector))
        assert loaded == vector
