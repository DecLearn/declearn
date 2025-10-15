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

"""Fed-FairGrad algorithm controllers and utils.

Introduction
------------
This module provides with an implementation of Fed-FairGrad,
a yet-to-be-published algorithm that adapts the FairGrad [1]
algorithm to the federated learning setting.

FairGrad aims at minimizing the training loss of a model under
group-fairness constraints, with an optional epsilon tolerance.
It relies on reweighting the loss using weights that are based
on sensitive groups, and are updated throughout training based
on estimates of the current fairness of the trained model.

Fed-FairGrad formulates the same problem, and adjusts client-wise
weights based on the repartition of group-wise data across clients.
In its current version, the algorithm has fixed weights across local
training steps that are taken between model aggregation steps, while
the weights are updated based on robust estimates of the aggregated
model's fairness on the federated training data.

This algorithm is designed for settings where a classifier is trained
over data with any number of categorical sensitive attributes. It may
evolve as more theoretical and/or empirical results are obtained as to
its performance (both in terms of utility and fairness).

Controllers
-----------
* [FairgradControllerClient]
[declearn.fairness.fairgrad.FairgradControllerClient]:
    Client-side controller to implement Fed-FairGrad.
* [FairgradControllerServer]
[declearn.fairness.fairgrad.FairgradControllerServer]:
    Server-side controller to implement Fed-FairGrad.

Backend
-------
* [FairgradWeightsController]
[declearn.fairness.fairgrad.FairgradWeightsController]:
    Controller to implement Faigrad optimization constraints.

Messages
--------
* [FairgradOkay][declearn.fairness.fairgrad.FairgradOkay]
* [FairgradWeights][declearn.fairness.fairgrad.FairgradWeights]


References
----------
- [1]
    Maheshwari & Perrot (2023).
    FairGrad: Fairness Aware Gradient Descent.
    https://openreview.net/forum?id=0f8tU3QwWD
"""

from ._client import FairgradControllerClient
from ._messages import (
    FairgradOkay,
    FairgradWeights,
)
from ._server import FairgradControllerServer, FairgradWeightsController
