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

"""FedAvg-like mean-aggregation class."""

from typing import Any, Dict

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
    ) -> None:
        """Instantiate an averaging aggregator.

        Parameters
        ----------
        steps_weighted:
            Whether to conduct a weighted averaging of local model
            updates based on local numbers of training steps.
        """
        self.steps_weighted = steps_weighted

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {
            "steps_weighted": self.steps_weighted,
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
