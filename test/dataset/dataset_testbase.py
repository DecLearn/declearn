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

"""Unit tests for instances of 'declearn.dataset.Dataset'"""

from abc import abstractmethod
from dataclasses import asdict

import numpy as np

from declearn.dataset import Dataset
from declearn.test_utils import assert_batch_equal, to_numpy


class DatasetTestToolbox:
    """TestCase fixture-provider protocol."""

    # pylint: disable=too-few-public-methods

    framework: str

    data, label, weights = (
        np.concatenate([np.ones((2, 2)), np.zeros((2, 2))]),
        np.array([1, 2, 3, 4]),
        np.ones(4),
    )

    @abstractmethod
    def get_dataset(self) -> Dataset:
        """Convert the test data into a framework-specific dataset"""


class DatasetTestSuite:
    """Base tests for declearn Dataset abstract methods."""

    def test_generate_batches_batchsize(self, toolbox: DatasetTestToolbox):
        """Test batch_size argument to test_generate_batches method"""
        expected = [
            (np.ones((2, 2)), np.array([1, 2]), np.array([1.0, 1.0])),
            (np.zeros((2, 2)), np.array([3, 4]), np.array([1.0, 1.0])),
        ]
        result = toolbox.get_dataset().generate_batches(2)
        assert_batch_equal(result, expected, toolbox.framework)

    def test_generate_batches_shuffle(self, toolbox: DatasetTestToolbox):
        """Test the shuffle argument of the generate_batches method"""
        excluded = range(1, 5)
        result = toolbox.get_dataset().generate_batches(1, shuffle=True)
        assert any(
            (
                to_numpy(res[1], toolbox.framework)[0] != excluded[i]
                for i, res in enumerate(result)
            )
        )

    def test_generate_batches_remainder(self, toolbox: DatasetTestToolbox):
        """Test the drop_remainder argument of the generate_batches method"""
        # drop_remainder = True
        expected = [
            (
                np.concatenate([np.ones((2, 2)), np.zeros((1, 2))]),
                np.array([1, 2, 3]),
                np.ones(3),
            )
        ]
        result = toolbox.get_dataset().generate_batches(3)
        assert_batch_equal(result, expected, toolbox.framework)
        # drop_remainder = False
        expected = [
            (
                np.concatenate([np.ones((2, 2)), np.zeros((1, 2))]),
                np.array([1, 2, 3]),
                np.ones(3),
            ),
            (np.array([[0, 0]]), np.array([4]), np.array([1.0])),
        ]
        result = toolbox.get_dataset().generate_batches(
            3, drop_remainder=False
        )
        assert_batch_equal(result, expected, toolbox.framework)

    def test_generate_batches_replacement(self, toolbox: DatasetTestToolbox):
        """Test the 'replacement' argument of 'generate_batches'."""
        # Generate batches with replacement, 4 times (due to small dataset).
        dataset = toolbox.get_dataset()
        batches = [
            next(dataset.generate_batches(4, shuffle=True, replacement=True))
            for _ in range(4)
        ]
        # Verify that in at least one case, there are repeated samples.
        assert any(
            len(np.unique(to_numpy(batch[1], toolbox.framework))) < 4
            for batch in batches
        )

    def test_generate_batches_poisson(self, toolbox: DatasetTestToolbox):
        """Test the 'poisson' argument of 'generate_batches'."""
        # Generate poisson-sampled batches, 4 times (due to small dataset).
        dataset = toolbox.get_dataset()
        batches = [
            next(dataset.generate_batches(2, poisson=True)) for _ in range(4)
        ]
        b_sizes = [
            to_numpy(x, framework=toolbox.framework).shape[0]
            for x, *_ in batches
        ]
        # Assert that Poisson sampling results in varying-size batches.
        assert len(set(b_sizes)) > 1

    def test_get_data_specs(self, toolbox: DatasetTestToolbox):
        """Test the get_data_spec method"""
        specs = toolbox.get_dataset().get_data_specs()
        assert asdict(specs)["n_samples"] == 4
