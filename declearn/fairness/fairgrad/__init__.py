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

"""Fed-FairGrad algorithm controllers and utils.

Introduction
------------
This module provides with an implementation of Fed-FairGrad, a work-
in-progress algorithm that aims at adapting the FairGrad algorithm
(introduced by Maheshwari and Perrot (2022)) to the federated setting.

FairGrad formulates an optimization problem that aims at maximizing a
group-fairness function while minimizing the overall loss of a model.
Its solving relies on introducing sensitive-group-wise weights, that
are updated throughout the training based on estimates of the current
model's fairness on the training data.

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

Messages
--------
* [FairgradSetupQuery][declearn.fairness.fairgrad.FairgradSetupQuery]:
    Message for server-emitted Fed-FairGrad setup queries.
* [FairgradWeights][declearn.fairness.fairgrad.FairgradWeights]:
    Message for server-emitted (Fed-)FairGrad loss weights sharing.
"""

from ._messages import (
    FairgradSetupQuery,
    FairgradWeights,
)
from ._client import FairgradControllerClient
from ._server import FairgradControllerServer
