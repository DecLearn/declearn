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

"""Unit tests for 'declearn.fairness.api.FairnessMetricsComputer'."""

from typing import Optional
from unittest import mock

import numpy as np
import pytest

from declearn.dataset import Dataset
from declearn.fairness.api import FairnessDataset, FairnessMetricsComputer
from declearn.metrics import MeanMetric, MetricSet
from declearn.model.api import Model

N_BATCHES = 8
GROUPS = [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.fixture(name="dataset")
def dataset_fixture() -> FairnessDataset:
    """Mock FairnessDataset providing fixture."""
    # Set up a mock FairnessDataset.
    dataset = mock.create_autospec(FairnessDataset, instance=True)
    dataset.get_sensitive_group_definitions.return_value = GROUPS.copy()
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
            group=GROUPS[0],
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
        subset = computer.g_data[GROUPS[0]]
        subset.generate_batches.assert_called_once_with(  # type: ignore
            batch_size=8, shuffle=n_batch is not None, drop_remainder=False
        )

    def test_setup_accuracy_metric(
        self,
        dataset: FairnessDataset,
    ) -> None:
        """Verify that 'setup_accuracy_metric' works properly."""
        # Set up an accuracy metric with an arbitrary threshold.
        computer = FairnessMetricsComputer(dataset)
        model = mock.create_autospec(Model, instance=True)
        metric = computer.setup_accuracy_metric(model, thresh=0.65)
        # Verify that the metric performs expected comptuations.
        assert isinstance(metric, MeanMetric)
        metric.update(y_true=np.ones(4), y_pred=np.ones(4) * 0.7)
        assert metric.get_result()[metric.name] == 1.0
        metric.reset()
        metric.update(y_true=np.ones(4), y_pred=np.ones(4) * 0.6)
        assert metric.get_result()[metric.name] == 0.0

    def test_setup_loss_metric(
        self,
        dataset: FairnessDataset,
    ) -> None:
        """Verify that 'setup_loss_metric' works properly."""
        # Set up an accuracy metric with an arbitrary threshold.
        computer = FairnessMetricsComputer(dataset)
        model = mock.create_autospec(Model, instance=True)

        def mock_loss_function(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            s_wght: Optional[np.ndarray] = None,
        ) -> np.ndarray:
            """Mock model loss function."""
            # API-defined signature; pylint: disable=unused-argument
            return np.ones_like(y_pred) * 0.05

        model.loss_function.side_effect = mock_loss_function
        metric = computer.setup_loss_metric(model)
        # Verify that the metric performs expected comptuations.
        assert isinstance(metric, MeanMetric)
        metric.update(y_true=np.ones(4), y_pred=np.ones(4))
        assert metric.get_result()[metric.name] == 0.05
        model.loss_function.assert_called_once()

    def test_compute_groupwise_metrics(
        self,
        dataset: FairnessDataset,
    ) -> None:
        """Test the 'compute_groupwise_metrics' method."""
        # Set up mock objects and run (mocked) computations.
        computer = FairnessMetricsComputer(dataset)
        model = mock.create_autospec(Model, instance=True)
        metrics = [
            computer.setup_accuracy_metric(model),
            computer.setup_loss_metric(model),
        ]
        with mock.patch.object(
            computer, "compute_metrics_over_sensitive_group"
        ) as patch_compute_metrics_over_sensitive_group:
            results = computer.compute_groupwise_metrics(
                metrics=metrics,
                model=model,
                batch_size=16,
                n_batch=32,
            )
        # Verify that outputs have expected types and dict keys.
        assert isinstance(results, dict)
        assert set(results) == {metric.name for metric in metrics}
        for m_dict in results.values():
            assert isinstance(m_dict, dict)
            assert set(m_dict) == set(GROUPS)
            assert all(isinstance(value, float) for value in m_dict.values())
        # Verify that expected calls occured.
        patch_compute_metrics_over_sensitive_group.assert_has_calls(
            [mock.call(group, mock.ANY, model, 16, 32) for group in GROUPS],
            any_order=True,
        )

    def test_scale_metrics_by_sample_counts(
        self,
    ) -> None:
        """Test that 'scale_metrics_by_sample_counts' works properly."""
        # Set up a mock FairnessDataset and wrap it up with a metrics computer.
        dataset = mock.create_autospec(FairnessDataset, instance=True)
        dataset.get_sensitive_group_definitions.return_value = GROUPS
        dataset.get_sensitive_group_counts.return_value = {
            group: idx for idx, group in enumerate(GROUPS, start=1)
        }
        computer = FairnessMetricsComputer(dataset)
        # Test the 'scale_metrics_by_sample_counts' method.
        metrics = {
            group: float(idx) for idx, group in enumerate(GROUPS, start=1)
        }
        metrics = computer.scale_metrics_by_sample_counts(metrics)
        expected = {
            group: float(idx**2) for idx, group in enumerate(GROUPS, start=1)
        }
        assert metrics == expected
