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

"""FairFed-specific Aggregator subclass."""

from declearn.aggregator import Aggregator, ModelUpdates
from declearn.model.api import Vector

__all__ = [
    "FairfedAggregator",
]


class FairfedAggregator(Aggregator, register=False):
    """Fairfed-specific Aggregator using arbitrary averaging weights."""

    name = "fairfed"

    def __init__(
        self,
        beta: float = 1.0,
    ) -> None:
        """Instantiate the Fairfed-specific weight averaging aggregator.

        Parameters
        ----------
        beta:
            Hyper-parameter controlling the magnitude of averaging weights'
            updates across rounds.
        """
        self.beta = beta
        self._weight = 1.0

    def initialize_local_weight(
        self,
        n_samples: int,
    ) -> None:
        """Initialize the local averaging weight based on dataset size."""
        self._weight = n_samples

    def update_local_weight(
        self,
        delta_loc: float,
        delta_avg: float,
    ) -> None:
        """Update the local averaging weight based on fairness measures.

        Parameters
        ----------
        delta_loc:
            Absolute difference between the local and global fairness values.
        delta_avg:
            Average of `delta_loc` values across all clients.
        """
        update = self.beta * (delta_loc - delta_avg)
        self._weight -= update

    def prepare_for_sharing(
        self,
        updates: Vector,
        n_steps: int,
    ) -> ModelUpdates:
        updates = updates * self._weight
        return ModelUpdates(updates=updates, weights=self._weight)

    def finalize_updates(
        self,
        updates: ModelUpdates,
    ) -> Vector:
        return updates.updates / updates.weights
