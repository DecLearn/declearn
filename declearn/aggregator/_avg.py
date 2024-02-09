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

import warnings
from typing import Any, Dict, Optional


from declearn.aggregator._api import Aggregator, ModelUpdates
from declearn.model.api import Vector

__all__ = [
    "AveragingAggregator",
]


class AveragingAggregator(Aggregator[ModelUpdates]):
    """Average-based-aggregation Aggregator subclass.

    This class implements local updates' averaging, with optional
    client-based and/or number-of-training-steps-based weighting.

    It may therefore be used to implement FedAvg and derivatives
    that use simple weighting schemes.
    """

    name = "averaging"

    def __init__(
        self,
        steps_weighted: bool = True,
        client_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Instantiate an averaging aggregator.

        Parameters
        ----------
        steps_weighted:
            Whether to conduct a weighted averaging of local model
            updates based on local numbers of training steps.
        client_weights:
            DEPRECATED - this argument no longer affects computations,
            save when using the deprecated 'aggregate' method.
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
        if client_weights:  # pragma: no cover
            warnings.warn(
                f"'client_weights' argument to '{self.__class__.__name__}' was"
                " deprecated in DecLearn v2.4 and is no longer used, saved by"
                " the deprecated 'aggregate' method. It will be removed in"
                " DecLearn v2.6 and/or v3.0.",
                DeprecationWarning,
            )

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {
            "steps_weighted": self.steps_weighted,
            "client_weights": self.client_weights,
        }

    def prepare_for_sharing(
        self,
        updates: Vector,
        n_steps: int,
    ) -> ModelUpdates:
        if self.steps_weighted:
            updates = updates * n_steps
            weights = n_steps
        else:
            weights = 1
        return ModelUpdates(updates, weights)

    def finalize_updates(
        self,
        updates: ModelUpdates,
    ) -> Vector:
        return updates.updates / updates.weights

    def aggregate(
        self,
        updates: Dict[str, Vector],
        n_steps: Dict[str, int],
    ) -> Vector:
        # Make use of 'client_weights' as part of this DEPRECATED method.
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=DeprecationWarning)
            weights = self.compute_client_weights(updates, n_steps)
        steps_weighted = self.steps_weighted
        try:
            self.steps_weighted = True
            return super().aggregate(updates, weights)  # type: ignore
        finally:
            self.steps_weighted = steps_weighted

    def compute_client_weights(
        self,
        updates: Dict[str, Vector],
        n_steps: Dict[str, int],
    ) -> Dict[str, float]:
        """Compute weights to use when averaging a given set of updates.

        This method is DEPRECATED as of DecLearn v2.4.
        It will be removed in DecLearn 2.6 and/or 3.0.

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
        warnings.warn(
            f"'{self.__class__.__name__}.compute_client_weights' was"
            " deprecated in DecLearn v2.4. It will be removed in DecLearn"
            " v2.6 and/or v3.0.",
            DeprecationWarning,
        )
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
