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

"""Unit tests for 'declearn.dataset.tensorflow.TensorflowDataset'."""

import os
import warnings

# pylint: disable=duplicate-code
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


class TensorflowDatasetTestToolbox(DatasetTestToolbox):
    """Toolbox for TensorflowDataset."""

    # pylint: disable=too-few-public-methods

    framework = "tensorflow"

    seed = 20230731

    def get_dataset(self) -> Dataset:
        dst = tf.data.Dataset.from_tensor_slices(
            (self.data, self.label, self.weights)
        )
        return TensorflowDataset(dst, seed=self.seed)


@pytest.fixture(name="toolbox")
def fixture_dataset() -> DatasetTestToolbox:
    """Fixture to access a TensorflowTestCase."""
    return TensorflowDatasetTestToolbox()


class TestTensorflowDataset(DatasetTestSuite):
    """Unit tests for `declearn.dataset.tensorflow.TensorflowDataset`."""
