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

"""Unit tests for 'declearn.dataset.tensorflow.TensorflowDataset'."""

import os
import warnings

# pylint: disable=duplicate-code
import numpy as np
import pytest

try:
    with warnings.catch_warnings():  # silence tensorflow import-time warnings
        warnings.simplefilter("ignore")
        import tensorflow as tf  # type: ignore
except ModuleNotFoundError:
    pytest.skip("TensorFlow is unavailable", allow_module_level=True)
# pylint: enable=duplicate-code

from declearn.dataset import Dataset
from declearn.dataset.tensorflow import TensorflowDataset
from declearn.test_utils import make_importable

# relative imports from `dataset_testbase`
with make_importable(os.path.dirname(__file__)):
    from dataset_testbase import DatasetTestSuite, DatasetTestToolbox


SEED = 20230731


class TensorflowDatasetTestToolbox(DatasetTestToolbox):
    """Toolbox for TensorflowDataset."""

    # pylint: disable=too-few-public-methods

    framework = "tensorflow"

    def get_dataset(self) -> Dataset:
        dst = tf.data.Dataset.from_tensor_slices(
            (self.data, self.label, self.weights)
        )
        return TensorflowDataset(dst, seed=SEED)


@pytest.fixture(name="toolbox")
def fixture_dataset() -> DatasetTestToolbox:
    """Fixture to access a TensorflowTestCase."""
    return TensorflowDatasetTestToolbox()


@pytest.fixture(name="sentences_dataset")
def fixture_sentences_dataset() -> tf.data.Dataset:
    """Fixture to setup a tf.data.Dataset yielding sequences of tokens."""
    # Generate 32 variable-size sequences of int (tokenized sentences).
    rng = np.random.default_rng(seed=SEED)
    samples = [rng.choice(128, size=rng.choice(100) + 1) for _ in range(32)]
    # Wrap this data into a tensorflow dataset and return it.
    tf_data = tf.data.Dataset.from_generator(
        lambda: iter(samples),
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
    return tf_data


class TestTensorflowDatasetBase(DatasetTestSuite):
    """Unit tests for `declearn.dataset.tensorflow.TensorflowDataset`."""

    def test_generate_padded_batches(
        self,
        sentences_dataset: tf.data.Dataset,
    ) -> None:
        """Test 'generate_batches' with samples that require padding."""
        dataset = TensorflowDataset(
            sentences_dataset, batch_mode="padded", seed=SEED
        )
        batches = list(dataset.generate_batches(batch_size=8, poisson=False))
        # Verify that there are 4 batches, with inputs of shape (8, [1-100]).
        assert len(batches) == 4
        shapes = [inputs.shape for inputs, _, _ in batches]  # type: ignore
        assert all(shp[0] == 8 for shp in shapes)  # single batch size
        assert len(set(shp[1] for shp in shapes)) > 1  # various seq length
        assert all(1 <= shp[1] <= 100 for shp in shapes)

    def test_generate_padded_batches_with_poisson(
        self,
        sentences_dataset: tf.data.Dataset,
    ) -> None:
        """Test 'generate_batches(poisson=True)' with samples to be padded."""
        dataset = TensorflowDataset(
            sentences_dataset, batch_mode="padded", seed=SEED
        )
        batches = list(dataset.generate_batches(batch_size=8, poisson=True))
        # Verify that there are 4 batches, with inputs of shape (?, [1-100]).
        assert len(batches) == 4
        shapes = [inputs.shape for inputs, _, _ in batches]  # type: ignore
        assert len(set(shp[0] for shp in shapes)) > 1  # various batch size
        assert len(set(shp[1] for shp in shapes)) > 1  # various seq length
        assert all(1 <= shp[1] <= 100 for shp in shapes)

    def test_generate_ragged_batches(
        self,
        sentences_dataset: tf.data.Dataset,
    ) -> None:
        """Test 'generate_batches' with samples that require ragging."""
        dataset = TensorflowDataset(
            sentences_dataset, batch_mode="ragged", seed=SEED
        )
        batches = list(dataset.generate_batches(batch_size=8, poisson=False))
        # Verify that there are 4 ragged batches, with inputs of shape (8, ?).
        assert len(batches) == 4
        assert all(isinstance(x, tf.RaggedTensor) for x, _, _ in batches)
        shapes = [inputs.shape for inputs, _, _ in batches]  # type: ignore
        assert all(shp[0] == 8 for shp in shapes)  # single batch size
        assert all(shp[1] is None for shp in shapes)  # ragged seq length

    def test_generate_ragged_batches_with_poisson(
        self,
        sentences_dataset: tf.data.Dataset,
    ) -> None:
        """Test 'generate_batches(poisson=True)' with samples to be ragged."""
        dataset = TensorflowDataset(
            sentences_dataset, batch_mode="ragged", seed=SEED
        )
        batches = list(dataset.generate_batches(batch_size=8, poisson=True))
        # Verify that there are 4 ragged batches, with inputs of shape (?, ?).
        assert len(batches) == 4
        assert all(isinstance(x, tf.RaggedTensor) for x, _, _ in batches)
        shapes = [inputs.shape for inputs, _, _ in batches]  # type: ignore
        assert len(set(shp[0] for shp in shapes)) > 1  # various batch size
        assert all(shp[1] is None for shp in shapes)  # ragged seq length
