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

"""FedAvg-like mean-aggregation class."""

from typing import Any, ClassVar, Dict, Optional

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.aggregator._api import Aggregator
from declearn.model.api import Vector

__all__ = [
    "AveragingAggregator",
]


class AveragingAggregator(Aggregator):
    """Average-based-aggregation Aggregator subclass.

    This class implements local updates' averaging, with optional
    client-based and/or number-of-training-steps-based weighting.

    It may therefore be used to implement FedAvg and derivatives
    that use simple weighting schemes.
    """

    name: ClassVar[str] = "averaging"

    def __init__(
        self,
        steps_weighted: bool = True,
        client_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Instantiate an averaging aggregator.

        Parameters
        ----------
        steps_weighted: bool, default=True
            Whether to weight updates based on the number of optimization
            steps taken by the clients (relative to one another).
        client_weights: dict[str, float] or None, default=None
            Optional dict of client-wise base weights to use.
            If None, homogeneous base weights are used.

        Notes
        -----
        * One may specify `client_weights` and use `steps_weighted=True`.
          In that case, the product of the client's base weight and their
          number of training steps taken will be used (and unit-normed).
        * One may use incomplete `client_weights`. In that case, unknown-
          clients' base weights will be set to 1.
        """
        self.steps_weighted = steps_weighted
        self.client_weights = client_weights or {}

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {
            "steps_weighted": self.steps_weighted,
            "client_weights": self.client_weights,
        }

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        return cls(**config)

    def aggregate(
        self,
        updates: Dict[str, Vector],
        n_steps: Dict[str, int],
    ) -> Vector:
        if not updates:
            raise TypeError("Cannot aggregate an empty set of updates.")
        weights = self.compute_client_weights(updates, n_steps)
        agg = sum(grads * weights[client] for client, grads in updates.items())
        return agg  # type: ignore

    def compute_client_weights(
        self,
        updates: Dict[str, Vector],
        n_steps: Dict[str, int],
    ) -> Dict[str, float]:
        """Compute weights to use when averaging a given set of updates.

        Parameters
        ----------
        updates: dict[str, Vector]
            Client-wise updates, as a dictionary with clients' names as
            string keys and updates as Vector values.
        n_steps: dict[str, int]
            Client-wise number of local training steps performed during
            the training round having produced the updates.

        Returns
        -------
        weights: dict[str, float]
            Client-wise updates-averaging weights, suited to the input
            parameters and normalized so that they sum to 1.
        """
        if self.steps_weighted:
            weights = {
                client: steps * self.client_weights.get(client, 1.0)
                for client, steps in n_steps.items()
            }
        else:
            weights = {
                client: self.client_weights.get(client, 1.0)
                for client in updates
            }
        total = sum(weights.values())
        return {client: weight / total for client, weight in weights.items()}
