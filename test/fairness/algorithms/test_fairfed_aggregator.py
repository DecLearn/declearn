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

"""Unit tests for FairFed-specific Aggregator subclass."""

from unittest import mock

from declearn.fairness.fairfed import FairfedAggregator
from declearn.model.api import Vector


class TestFairfedAggregator:
    """Unit tests for 'declearn.fairness.fairfed.FairfedAggregator'."""

    def test_init_beta(self) -> None:
        """Test that the 'beta' parameter is properly assigned."""
        beta = mock.create_autospec(float, instance=True)
        aggregator = FairfedAggregator(beta=beta)
        assert aggregator.beta is beta

    def test_prepare_for_sharing_initial(self) -> None:
        """Test that 'prepare_for_sharing' has expected outputs at first."""
        # Set up an uninitialized aggregator and prepare mock updates.
        aggregator = FairfedAggregator(beta=1.0)
        updates = mock.create_autospec(Vector, instance=True)
        model_updates = aggregator.prepare_for_sharing(updates, n_steps=10)
        # Verify that outputs match expectations.
        updates.__mul__.assert_called_once_with(1.0)
        assert model_updates.updates is updates.__mul__.return_value
        assert model_updates.weights == 1.0

    def test_initialize_local_weight(self) -> None:
        """Test that 'initialize_local_weight' works properly."""
        # Set up an aggregator, initialize it and prepare mock updates.
        n_samples = 100
        aggregator = FairfedAggregator(beta=1.0)
        aggregator.initialize_local_weight(n_samples=n_samples)
        updates = mock.create_autospec(Vector, instance=True)
        model_updates = aggregator.prepare_for_sharing(updates, n_steps=10)
        # Verify that outputs match expectations.
        updates.__mul__.assert_called_once_with(n_samples)
        assert model_updates.updates is updates.__mul__.return_value
        assert model_updates.weights == n_samples

    def test_update_local_weight(self) -> None:
        """Test that 'update_local_weight' works properly."""
        # Set up a FairFed aggregator and initialize it.
        n_samples = 100
        aggregator = FairfedAggregator(beta=0.1)
        aggregator.initialize_local_weight(n_samples=n_samples)
        # Perform a local wiehgt update with arbitrary values.
        aggregator.update_local_weight(delta_loc=2.0, delta_avg=5.0)
        # Verify that updates have expected weight.
        updates = mock.create_autospec(Vector, instance=True)
        expectw = n_samples - 0.1 * (2.0 - 5.0)  # w_0 - beta * diff_delta
        model_updates = aggregator.prepare_for_sharing(updates, n_steps=10)
        updates.__mul__.assert_called_once_with(expectw)
        assert model_updates.updates is updates.__mul__.return_value
        assert model_updates.weights == expectw

    def test_finalize_updates(self) -> None:
        """Test that 'finalize_updates' works as expected."""
        # Set up a FairFed aggregator and initialize it.
        n_samples = 100
        aggregator = FairfedAggregator(beta=0.1)
        aggregator.initialize_local_weight(n_samples=n_samples)
        # Prepare, then finalize updates.
        updates = mock.create_autospec(Vector, instance=True)
        output = aggregator.finalize_updates(
            aggregator.prepare_for_sharing(updates, n_steps=mock.MagicMock())
        )
        assert output == (updates * n_samples / n_samples)
