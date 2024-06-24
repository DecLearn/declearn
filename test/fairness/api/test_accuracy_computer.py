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

"""Unit tests for 'declearn.fairness.api.FairnessMetricsComputer'."""

from typing import Optional
from unittest import mock

import pytest

from declearn.dataset import Dataset
from declearn.fairness.api import FairnessMetricsComputer, FairnessDataset
from declearn.metrics import MetricSet
from declearn.model.api import Model


N_BATCHES = 8


@pytest.fixture(name="dataset")
def dataset_fixture() -> FairnessDataset:
    """Mock FairnessDataset providing fixture."""
    # Set up a mock FairnessDataset.
    groups = [(0, 0), (0, 1), (1, 0), (1, 1)]
    dataset = mock.create_autospec(FairnessDataset, instance=True)
    dataset.get_sensitive_group_definitions.return_value = groups
    # Set up a mock Dataset.
    subdataset = mock.create_autospec(Dataset, instance=True)
    batches = [mock.MagicMock() for _ in range(N_BATCHES)]
    subdataset.generate_batches.return_value = iter(batches)
    # Have the FairnessDataset return the Dataset for any group.
    dataset.get_sensitive_group_subset.return_value = subdataset
    return dataset


class TestFairnessMetricsComputer:
    """Unit tests for 'declearn.fairness.api.FairnessMetricsComputer'."""

    @pytest.mark.parametrize("n_batch", [None, 4, 12])
    def test_compute_metrics_over_sensitive_groups(
        self,
        dataset: FairnessDataset,
        n_batch: Optional[int],
    ) -> None:
        """Test the 'compute_metrics_over_sensitive_groups' method."""
        # Set up mock objects and run (mocked) computations.
        computer = FairnessMetricsComputer(dataset)
        metrics = mock.create_autospec(MetricSet, instance=True)
        model = mock.create_autospec(Model, instance=True)
        mock_pred = (mock.MagicMock(), mock.MagicMock(), None)
        model.compute_batch_predictions.return_value = mock_pred
        results = computer.compute_metrics_over_sensitive_group(
            group=(0, 0),
            metrics=metrics,
            model=model,
            batch_size=8,
            n_batch=n_batch,
        )
        # Verify that expected (mocked) computations happened.
        expected_nbatches = min(n_batch or N_BATCHES, N_BATCHES)
        assert results is metrics.get_result.return_value
        metrics.reset.assert_called_once()
        assert metrics.update.call_count == expected_nbatches
        assert model.compute_batch_predictions.call_count == expected_nbatches
        subset = computer.g_data[(0, 0)]
        subset.generate_batches.assert_called_once_with(  # type: ignore
            batch_size=8, shuffle=n_batch is not None, drop_remainder=False
        )
