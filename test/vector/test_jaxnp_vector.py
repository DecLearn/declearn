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

"""Unit tests for 'declearn.model.haiku.JaxNumpyVector'."""

import os

import pytest

# pylint: disable=duplicate-code

try:
    import jax
except ModuleNotFoundError:
    pytest.skip("jax and/or haiku are unavailable", allow_module_level=True)

# pylint: enable=duplicate-code

from declearn.model.haiku import JaxNumpyVector
from declearn.model.haiku.utils import select_device
from declearn.model.sklearn import NumpyVector
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(__file__)):
    from vector_testing import VectorFactory, VectorTestSuite


# Enable float64 support.
jax.config.update("jax_enable_x64", True)


class JaxNumpyVectorFactory(VectorFactory):
    """Factory for RNG-seeded fixed-spec JaxNumpyVector instances."""

    framework = "jax"
    vector_cls = JaxNumpyVector

    def make_vector(
        self,
        seed: int = 0,
    ) -> JaxNumpyVector:
        values = self.make_values(seed)
        device = select_device(gpu=False)
        return JaxNumpyVector(
            {key: jax.device_put(val, device) for key, val in values.items()}
        )


@pytest.fixture(name="factory")
def fixture_factory() -> JaxNumpyVectorFactory:
    """Fixture providing with a JaxNumpyVectorFactory."""
    return JaxNumpyVectorFactory()


class TestJaxNumpyVector(VectorTestSuite):
    """Unit tests for JaxNumpyVector."""

    def test_sub_numpy_vector(
        self,
        factory: JaxNumpyVectorFactory,
    ) -> None:
        """Test subtracting a NumpyVector from a TorchVector."""
        jx_ref = factory.make_vector(seed=0)
        jx_vec = factory.make_vector(seed=1)
        np_vec = NumpyVector(factory.make_values(seed=1))
        result = jx_ref - np_vec
        expect = jx_ref - jx_vec
        assert isinstance(result, JaxNumpyVector)
        assert result == expect

    def test_rsub_numpy_vector(
        self,
        factory: JaxNumpyVectorFactory,
    ) -> None:
        """Test subtracting a TorchVector from a TorchVector."""
        jx_ref = factory.make_vector(seed=0)
        jx_vec = factory.make_vector(seed=1)
        np_vec = NumpyVector(factory.make_values(seed=1))
        result = np_vec - jx_ref
        expect = jx_vec - jx_ref
        assert isinstance(result, JaxNumpyVector)
        assert result == expect
