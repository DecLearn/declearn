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

"""Sum-aggregation Aggregator subclass."""

from declearn.aggregator._api import Aggregator, ModelUpdates
from declearn.model.api import Vector

__all__ = [
    "SumAggregator",
]


class SumAggregator(Aggregator[ModelUpdates]):
    """Sum-aggregation Aggregator subclass.

    This class implements the mere summation of client-wise model
    updates. It is therefore targetted at algorithms that perform
    some processing on model updates (e.g. via sample weights) so
    that mere summation is the proper way to recover gradients of
    the global model.
    """

    name = "sum"

    def prepare_for_sharing(
        self,
        updates: Vector,
        n_steps: int,
    ) -> ModelUpdates:
        return ModelUpdates(updates=updates, weights=1)

    def finalize_updates(
        self,
        updates: ModelUpdates,
    ) -> Vector:
        return updates.updates
