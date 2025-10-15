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

"""FairFed algorithm controllers and utils.

Introduction
------------
This module provides with an implementation of FairFed [1], an
algorithm that aims at enforcing fairness in a federated learning
setting by weighting client-wise model updates' averaging based on
differences between the global and local fairness of the (prior
version of the) shared model, using somewhat ad hoc discrepancy
metrics to summarize fairness as scalar values.

This algorithm was originally designed for settings where a binary
classifier is trained over data with a single binary sensitive
attribute, with the authors showcasing their generic formulas over
a limited set of group fairness definitions. DecLearn expands it to
a broader case, enabling the use of arbitrary fairness definitions
over data that may have non-binary and/or many sensitive attributes.
A 'strict' mode is made available to stick to the original paper.

Additionally, the algorithm's authors suggest combining it with other
mechanisms that aim at enforcing model fairness during local training
steps. At the moment, our implementation does not support this, but
future refactoring of routines that make up for a federated learning
process may enable doing so.

Controllers
-----------
* [FairfedControllerClient]
[declearn.fairness.fairfed.FairfedControllerClient]:
    Client-side controller to implement FairFed.
* [FairfedControllerServer]
[declearn.fairness.fairfed.FairfedControllerServer]:
    Server-side controller to implement FairFed.

Backend
-------
* [FairfedAggregator][declearn.fairness.fairfed.FairfedAggregator]:
    Fairfed-specific Aggregator using arbitrary averaging weights.
* [FairfedValueComputer][declearn.fairness.fairfed.FairfedValueComputer]:
    Fairfed-specific synthetic fairness value computer.

Messages
--------
* [FairfedDelta][declearn.fairness.fairfed.FairfedDelta]
* [FairfedDeltavg][declearn.fairness.fairfed.FairfedDeltavg]
* [FairfedFairness][declearn.fairness.fairfed.FairfedFairness]
* [FairfedOkay][declearn.fairness.fairfed.FairfedOkay]
* [SecaggFairfedDelta][declearn.fairness.fairfed.SecaggFairfedDelta]

References
----------
- [1]
    Ezzeldin et al. (2021).
    FairFed: Enabling Group Fairness in Federated Learning
    https://arxiv.org/abs/2110.00857
"""

from ._aggregator import FairfedAggregator
from ._client import FairfedControllerClient
from ._fairfed import FairfedValueComputer
from ._messages import (
    FairfedDelta,
    FairfedDeltavg,
    FairfedFairness,
    FairfedOkay,
    SecaggFairfedDelta,
)
from ._server import FairfedControllerServer
