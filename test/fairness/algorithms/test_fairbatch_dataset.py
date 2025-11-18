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

"""Unit tests for FairBatch dataset wrapper."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from declearn.fairness.api import FairnessDataset
from declearn.fairness.core import FairnessInMemoryDataset
from declearn.fairness.fairbatch import FairbatchDataset

COUNTS = {(0, 0): 30, (0, 1): 15, (1, 0): 35, (1, 1): 20}


class TestFairbatchDataset:
    """Unit tests for 'declearn.fairness.fairbatch.FairbatchDataset'."""

    def setup_mock_base_dataset(self) -> mock.Mock:
        """Return a mock FairnessDataset with arbitrary groupwise counts."""
        base = mock.create_autospec(FairnessDataset, instance=True)
        base.get_sensitive_group_definitions.return_value = list(COUNTS)
        base.get_sensitive_group_counts.return_value = COUNTS
        return base

    def test_wrapped_methods(self) -> None:
        """Test that API-defined methods are properly wrapped."""
        # Instantiate a FairbatchDataset wrapping a mock FairnessDataset.
        base = mock.create_autospec(FairnessDataset, instance=True)
        data = FairbatchDataset(base)
        # Test API-defined getters.
        assert data.get_data_specs() is base.get_data_specs.return_value
        assert data.get_sensitive_group_definitions() is (
            base.get_sensitive_group_definitions.return_value
        )
        assert data.get_sensitive_group_counts() is (
            base.get_sensitive_group_counts.return_value.copy()
        )
        group = mock.create_autospec(tuple, instance=True)
        assert data.get_sensitive_group_subset(group) is (
            base.get_sensitive_group_subset.return_value
        )
        base.get_sensitive_group_subset.assert_called_once_with(group)
        # Test API-defined setter.
        weights = mock.create_autospec(dict, instance=True)
        adjust_by_counts = mock.create_autospec(bool, instance=True)
        data.set_sensitive_group_weights(weights, adjust_by_counts)
        base.set_sensitive_group_weights.assert_called_once_with(
            weights, adjust_by_counts
        )

    def test_get_sampling_probabilities_initial(self) -> None:
        """Test 'get_sampling_probabilities' upon initialization."""
        # Instantiate a FairbatchDataset wrapping a mock FairnessDataset.
        base = self.setup_mock_base_dataset()
        data = FairbatchDataset(base)
        # Access initial sampling probabilities and verify their value.
        probas = data.get_sampling_probabilities()
        assert isinstance(probas, dict)
        assert probas.keys() == COUNTS.keys()
        assert all(isinstance(val, float) for val in probas.values())
        expected = {key: 1 / len(COUNTS) for key in COUNTS}
        assert probas == expected

    def test_set_sampling_probabilities_simple(self) -> None:
        """Test 'set_sampling_probabilities' with matching groups."""
        data = FairbatchDataset(base=self.setup_mock_base_dataset())
        # Assign arbitrary probabilities that match local groups.
        probas = {group: idx / 10 for idx, group in enumerate(COUNTS, 1)}
        data.set_sampling_probabilities(group_probas=probas)
        # Test that inputs were assigned.
        assert data.get_sampling_probabilities() == probas

    def test_set_sampling_probabilities_unnormalized(self) -> None:
        """Test 'set_sampling_probabilities' with un-normalized values."""
        data = FairbatchDataset(base=self.setup_mock_base_dataset())
        # Assign arbitrary probabilities that do not sum to 1.
        probas = {group: float(idx) for idx, group in enumerate(COUNTS, 1)}
        expect = {key: val / 10 for key, val in probas.items()}
        data.set_sampling_probabilities(group_probas=probas)
        # Test that inputs were cprrected, then assigned.
        assert data.get_sampling_probabilities() == expect

    def test_set_sampling_probabilities_superset(self) -> None:
        """Test 'set_sampling_probabilities' with unrepresented groups."""
        data = FairbatchDataset(base=self.setup_mock_base_dataset())
        # Assign arbitrary probabilities that cover a superset of local groups.
        probas = {group: idx / 10 for idx, group in enumerate(COUNTS, 1)}
        expect = probas.copy()
        probas[(2, 0)] = probas[(2, 1)] = 0.2
        data.set_sampling_probabilities(group_probas=probas)
        # Test that inputs were corrected, then assigned.
        assert data.get_sampling_probabilities() == expect

    def test_set_sampling_probabilities_invalid_values(self) -> None:
        """Test 'set_sampling_probabilities' with negative values."""
        probas = {group: float(idx) for idx, group in enumerate(COUNTS, -2)}
        data = FairbatchDataset(base=self.setup_mock_base_dataset())
        with pytest.raises(ValueError):
            data.set_sampling_probabilities(group_probas=probas)

    def test_set_sampling_probabilities_invalid_groups(self) -> None:
        """Test 'set_sampling_probabilities' with missing groups."""
        probas = {
            group: idx / 6 for idx, group in enumerate(list(COUNTS)[1:], 1)
        }
        data = FairbatchDataset(base=self.setup_mock_base_dataset())
        with pytest.raises(ValueError):
            data.set_sampling_probabilities(group_probas=probas)

    def setup_simple_dataset(self) -> FairbatchDataset:
        """Set up a simple FairbatchDataset with arbitrary data.

        Samples have a single feature, reflecting the sensitive
        group to which they belong.
        """
        samples = [
            sample
            for idx, (group, n_samples) in enumerate(COUNTS.items())
            for sample in [(group[0], group[1], idx)] * n_samples
        ]
        base = FairnessInMemoryDataset(
            data=pd.DataFrame(samples, columns=["target", "s_attr", "value"]),
            f_cols=["value"],
            target="target",
            s_attr=["s_attr"],
            sensitive_target=True,
        )
        # Wrap it up as a FairbatchDataset and assign arbitrary probabilities.
        return FairbatchDataset(base)

    def test_generate_batches_simple(self) -> None:
        """Test that 'generate_batches' has expected behavior."""
        # Setup a simple dataset and assign arbitrary sampling probabilities.
        data = self.setup_simple_dataset()
        data.set_sampling_probabilities(
            {group: idx / 10 for idx, group in enumerate(COUNTS, start=1)}
        )
        # Generate batches with a low batch size.
        # Verify that outputs match expectations.
        batches = list(data.generate_batches(batch_size=10))
        assert len(batches) == 10
        expect_x = np.array(
            [[idx] for idx in range(len(COUNTS)) for _ in range(idx + 1)]
        )
        expect_y = np.array(
            [lab for idx, (lab, _) in enumerate(COUNTS, 1) for _ in range(idx)]
        )
        for batch in batches:
            assert isinstance(batch, tuple) and (len(batch) == 3)
            assert isinstance(batch[0], np.ndarray)
            assert (batch[0] == expect_x).all()
            assert isinstance(batch[1], np.ndarray)
            assert (batch[1] == expect_y).all()
            assert batch[2] is None

    def test_generate_batches_large(self) -> None:
        """Test that 'generate_batches' has expected behavior."""
        # Setup a simple dataset and assign arbitrary sampling probabilities.
        data = self.setup_simple_dataset()
        data.set_sampling_probabilities(
            {group: idx / 10 for idx, group in enumerate(COUNTS, start=1)}
        )
        # Generate batches with a high batch size.
        # Verify that outputs match expectations.
        batches = list(data.generate_batches(batch_size=100))
        assert len(batches) == 1
        assert isinstance(batches[0][0], np.ndarray)
        assert isinstance(batches[0][1], np.ndarray)
        assert batches[0][2] is None
        expect_x = np.array(
            [
                [idx]
                for idx in range(len(COUNTS))
                for _ in range(10 * (idx + 1))
            ]
        )
        expect_y = np.array(
            [
                lab
                for idx, (lab, _) in enumerate(COUNTS, 1)
                for _ in range(idx * 10)
            ]
        )
        assert (batches[0][0] == expect_x).all()
        assert (batches[0][1] == expect_y).all()
