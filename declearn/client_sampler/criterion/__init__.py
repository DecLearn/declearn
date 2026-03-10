# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Criterion-based client sampling API and implementations.

`CriterionClientSampler` instances select clients based on a criterion score
that may be computed from client training replies and from the global model.

API Tools
---------

* [Criterion][declearn.client_sampler.criterion.Criterion]:
    Abstract base class defining an API for a client sampler criterion.
* [instantiate_criterion]\
[declearn.client_sampler.criterion.instantiate_criterion]
    Instantiate a `Criterion` from its specifications.

Concrete classes
----------------

* [CompositionCriterion]\
[declearn.client_sampler.criterion.CompositionCriterion]:
    Criterion subclass to perform composition of several criteria.
* [ConstantCriterion][declearn.client_sampler.criterion.ConstantCriterion]:
    Constant-valued Criterion subclass.
* [GradientNormCriterion]\
[declearn.client_sampler.criterion.GradientNormCriterion]:
    Gradients L2-norm Criterion subclass.
* [NormalizedDivCriterion]\
[declearn.client_sampler.criterion.NormalizedDivCriterion]:
    Normalized model divergence Criterion subclass.
* [TrainTimeCriterion][declearn.client_sampler.criterion.TrainTimeCriterion]:
    Last round training time Criterion subclass.
* [TrainTimeHistoryCriterion]\
[declearn.client_sampler.criterion.TrainTimeHistoryCriterion]:
    Training time history (across all past rounds) Criterion subclass.
"""

from ._api import (
    CompositionCriterion,
    ConstantCriterion,
    Criterion,
    instantiate_criterion,
)
from ._criteria import (
    GradientNormCriterion,
    NormalizedDivCriterion,
    TrainTimeCriterion,
    TrainTimeHistoryCriterion,
)

__all__ = [
    "CompositionCriterion",
    "ConstantCriterion",
    "Criterion",
    "GradientNormCriterion",
    "instantiate_criterion",
    "NormalizedDivCriterion",
    "TrainTimeCriterion",
    "TrainTimeHistoryCriterion",
]
