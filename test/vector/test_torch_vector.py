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

"""Unit tests for 'declearn.model.torch.TorchVector'."""

import os

import pytest

try:
    import torch
except ModuleNotFoundError:
    pytest.skip(reason="PyTorch is unavailable", allow_module_level=True)

from declearn.model.sklearn import NumpyVector
from declearn.model.torch import TorchVector
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(__file__)):
    from vector_testing import VectorFactory, VectorTestSuite


class TorchVectorFactory(VectorFactory):
    """Factory for RNG-seeded fixed-spec TorchVector instances."""

    framework = "torch"
    vector_cls = TorchVector

    def make_vector(
        self,
        seed: int = 0,
    ) -> TorchVector:
        values = self.make_values(seed)
        return TorchVector(
            # false-positive; pylint: disable=no-member
            {key: torch.from_numpy(val) for key, val in values.items()}
        )


@pytest.fixture(name="factory")
def fixture_factory() -> TorchVectorFactory:
    """Fixture providing with a TorchVectorFactory."""
    return TorchVectorFactory()


class TestTorchVector(VectorTestSuite):
    """Unit tests for TorchVector."""

    def test_sub_numpy_vector(
        self,
        factory: TorchVectorFactory,
    ) -> None:
        """Test subtracting a NumpyVector from a TorchVector."""
        pt_ref = factory.make_vector(seed=0)
        pt_vec = factory.make_vector(seed=1)
        np_vec = NumpyVector(factory.make_values(seed=1))
        result = pt_ref - np_vec
        expect = pt_ref - pt_vec
        assert isinstance(result, TorchVector)
        assert result == expect

    def test_rsub_numpy_vector(
        self,
        factory: TorchVectorFactory,
    ) -> None:
        """Test subtracting a TorchVector from a TorchVector."""
        pt_ref = factory.make_vector(seed=0)
        pt_vec = factory.make_vector(seed=1)
        np_vec = NumpyVector(factory.make_values(seed=1))
        result = np_vec - pt_ref
        expect = pt_vec - pt_ref
        assert isinstance(result, TorchVector)
        assert result == expect
