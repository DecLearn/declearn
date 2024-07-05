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

"""Fed-FairBatch / FedFB algorithm controllers and utils.

Introduction
------------
This module provides with a double-fold implementation of an adaptation
of the FairBatch [1] algorithm for federated learning. On the one hand,
the FedFB [2] algorithm is implemented, that both adapts FairBatch in a
straightforward manner and introduces changes in formulas compared with
the initial paper. On the other hand, the a custom algorithm deemed as
Fed-FairBatch is implemented, that is similar in intent to FedFB but
sticks to the raw FairBatch formulas.

FairBatch is a group-fairness-enforcing algorithm that relies on a
specific form of loss reweighting mediated by a specific batching
of samples for SGD steps. Namely, in FairBatch, batches are drawn
by concatenating group-wise sub-batches, the size of which is the
byproduct of the desired total batch size and group-wise sampling
probabilities, with the latter being updated throughout training
based on the current model's fairness.

Initially, FairBatch is designed for binary classification tasks
on data that have a single binary sensitive attribute. Both our
implementations currently stick to that setting, in spite of the
FedFB authors using a formalism that arguably extend formulas to
more generic categorical sensitive attribute(s) - which is not
tested in the paper.

Controllers
-----------
* [FairbatchControllerClient]
[declearn.fairness.fairbatch.FairgradControllerClient]:
    Client-side controller to implement Fed-FairBatch or FedFB.
* [FairbatchControllerServer]
[declearn.fairness.fairbatch.FairgradControllerServer]:
    Server-side controller to implement Fed-FairBatch or FedFB.

Backend
-------
* [FairbatchDataset][declearn.fairness.fairbatch.FairbatchDataset]:
    FairBatch-specific FairnessDataset subclass and wrapper.
* [FairbatchSamplingController]
[declearn.fairness.fairbatch.FairbatchSamplingController]:
    ABC to compute and update Fairbatch sampling probabilities.
* [setup_fairbatch_controller]
[declearn.fairness.fairbatch.setup_fairbatch_controller]:
    Instantiate a FairBatch sampling probabilities controller.
* [setup_fedfb_controller]
[declearn.fairness.fairbatch.setup_fedfb_controller]:
    Instantiate a FedFB sampling probabilities controller.

Messages
--------
* [FairbatchOkay][declearn.fairness.fairbatch.FairbatchOkay]
* [FairbatchSamplingProbas[
[declearn.fairness.fairbatch.FairbatchSamplingProbas]
"""

from ._messages import (
    FairbatchOkay,
    FairbatchSamplingProbas,
)
from ._sampling import (
    FairbatchSamplingController,
    setup_fairbatch_controller,
)
from ._fedfb import setup_fedfb_controller
from ._dataset import FairbatchDataset
from ._client import FairbatchControllerClient
from ._server import FairbatchControllerServer
