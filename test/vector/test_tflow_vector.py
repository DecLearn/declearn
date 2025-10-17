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

"""Unit tests for 'declearn.model.tensorflow.TensorflowVector'."""

import os

import pytest

try:
    import tensorflow as tf  # type: ignore
except ModuleNotFoundError:
    pytest.skip(reason="TensorFlow is unavailable", allow_module_level=True)

from declearn.model.sklearn import NumpyVector
from declearn.model.tensorflow import TensorflowVector
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(__file__)):
    from vector_testing import VectorFactory, VectorTestSuite


class TensorflowVectorFactory(VectorFactory):
    """Factory for RNG-seeded fixed-spec TensorflowVector instances."""

    framework = "tensorflow"
    vector_cls = TensorflowVector

    def make_vector(
        self,
        seed: int = 0,
        first_as_slices: bool = True,
    ) -> TensorflowVector:
        # Generate values and convert them to tensors.
        values = self.make_values(seed)
        tensor = {
            key: tf.convert_to_tensor(val) for key, val in values.items()
        }
        if first_as_slices:
            # Wrap the first one as IndexedSlices, made to be equivalent to
            # their dense counterpart. This is not very realistic, but enables
            # testing support for these structures as it enables comparing
            # outputs' values with numpy and other frameworks.
            tensor[self.names[0]] = tf.IndexedSlices(  # type: ignore
                values=tensor[self.names[0]],
                indices=tf.range(self.shapes[0][0]),
                dense_shape=tf.convert_to_tensor(self.shapes[0]),
            )
        return TensorflowVector(tensor)


@pytest.fixture(name="factory")
def fixture_factory() -> TensorflowVectorFactory:
    """Fixture providing with a TensorflowVectorFactory."""
    return TensorflowVectorFactory()


class TestTensorflowVector(VectorTestSuite):
    """Unit tests for TensorflowVector."""

    def test_sub_numpy_vector(
        self,
        factory: TensorflowVectorFactory,
    ) -> None:
        """Test subtracting a NumpyVector from a TensorflowVector."""
        tf_ref = factory.make_vector(seed=0, first_as_slices=False)
        tf_vec = factory.make_vector(seed=1, first_as_slices=False)
        np_vec = NumpyVector(factory.make_values(seed=1))
        result = tf_ref - np_vec
        expect = tf_ref - tf_vec
        assert isinstance(result, TensorflowVector)
        assert result == expect

    def test_rsub_numpy_vector(
        self,
        factory: TensorflowVectorFactory,
    ) -> None:
        """Test subtracting a TensorflowVector from a NumpyVector."""
        tf_ref = factory.make_vector(seed=0, first_as_slices=False)
        tf_vec = factory.make_vector(seed=1, first_as_slices=False)
        np_vec = NumpyVector(factory.make_values(seed=1))
        result = np_vec - tf_ref
        expect = tf_vec - tf_ref
        assert isinstance(result, TensorflowVector)
        assert result == expect

    def test_not_equal_dense_or_slices(
        self,
        factory: TensorflowVectorFactory,
    ) -> None:
        """Test that tensorflow Tensor and IndexedSlices are deemed unequal."""
        vec_a = factory.make_vector(seed=0, first_as_slices=True)
        vec_b = factory.make_vector(seed=0, first_as_slices=False)
        assert vec_a != vec_b
