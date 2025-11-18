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

"""Model updates aggregating API and implementations.

An `Aggregator` is typically meant to be used on a round-wise basis by
the orchestrating server of a centralized federated learning process,
to aggregate the client-wise model updated into a `Vector` that may then
be used as "gradients" by the server's `Optimizer` to update the global
model.

This declearn submodule provides with:

API tools
---------

* [Aggregator][declearn.aggregator.Aggregator]:
    Abstract base class defining an API for Vector aggregation.
* [ModelUpdates][declearn.aggregator.ModelUpdates]:
    Base dataclass for model updates' sharing and aggregation.
* [list_aggregators][declearn.aggregator.list_aggregators]:
    Return a mapping of registered Aggregator subclasses.


Concrete classes
----------------

* [AveragingAggregator][declearn.aggregator.AveragingAggregator]:
    Average-based-aggregation Aggregator subclass.
* [GradientMaskedAveraging][declearn.aggregator.GradientMaskedAveraging]:
    Gradient Masked Averaging Aggregator subclass.
* [SumAggregator][declearn.aggregator.SumAggregator]:
    Sum-aggregation Aggregator subclass.
"""

from ._api import Aggregator, ModelUpdates, list_aggregators
from ._avg import AveragingAggregator
from ._gma import GradientMaskedAveraging
from ._sum import SumAggregator
