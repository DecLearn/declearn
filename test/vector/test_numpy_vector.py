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

"""Unit tests for 'declearn.model.sklearn.NumpyVector'."""

import os

import pytest

from declearn.model.sklearn import NumpyVector
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(__file__)):
    from vector_testing import VectorFactory, VectorTestSuite


class NumpyVectorFactory(VectorFactory):
    """Factory for RNG-seeded fixed-spec NumpyVector instances."""

    framework = "numpy"
    vector_cls = NumpyVector

    def make_vector(
        self,
        seed: int = 0,
    ) -> NumpyVector:
        return NumpyVector(self.make_values(seed))


@pytest.fixture(name="factory")
def fixture_factory() -> NumpyVectorFactory:
    """Fixture providing with a NumpyVectorFactory."""
    return NumpyVectorFactory()


class TestNumpyVector(VectorTestSuite):
    """Unit tests for NumpyVector."""
