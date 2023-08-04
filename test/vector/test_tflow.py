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

"""Unit tests for 'declearn.model.tensorflow.TensorflowVector'."""

import os

import pytest

try:
    import tensorflow as tf  # type: ignore
except ModuleNotFoundError:
    pytest.skip(reason="TensorFlow is unavailable", allow_module_level=True)

from declearn.model.tensorflow import TensorflowVector
from declearn.utils import set_device_policy
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(__file__)):
    from vector_testing import VectorFactory, VectorTestSuite


set_device_policy(gpu=False)


class TensorflowVectorFactory(VectorFactory):
    """Factory for RNG-seeded fixed-spec TensorflowVector instances."""

    framework = "tensorflow"
    vector_cls = TensorflowVector

    def make_vector(
        self,
        seed: int = 0,
    ) -> TensorflowVector:
        values = self.make_values(seed)
        return TensorflowVector(
            {key: tf.convert_to_tensor(val) for key, val in values.items()}
        )


@pytest.fixture(name="factory")
def fixture_factory() -> TensorflowVectorFactory:
    """Fixture providing with a TensorflowVectorFactory."""
    return TensorflowVectorFactory()


class TestTensorflowVector(VectorTestSuite):
    """Unit tests for TensorflowVector."""
