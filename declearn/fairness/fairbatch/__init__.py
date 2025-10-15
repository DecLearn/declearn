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

"""Fed-FairBatch / FedFB algorithm controllers and utils.

Introduction
------------
FairBatch [1] is a group-fairness-enforcing algorithm that relies
on a specific form of loss reweighting mediated via the batching
of samples for SGD steps. Namely, in FairBatch, batches are drawn
by concatenating group-wise sub-batches, the size of which is the
byproduct of the desired total batch size and group-wise sampling
probabilities, with the latter being updated throughout training
based on the measured fairness of the current model.

This module provides with a double-fold adaptation of FairBatch to
the federated learning setting. On the one hand, a straightforward
adaptation using the law of total probability is proposed, that is
not based on any published paper. On the other hand, the FedFB [2]
algorithm is implemented, which adapts FairBatch in a similar way
but further introduces changes in formulas compared with the base
paper. Both variants are available via a unique pair of classes,
with a boolean switch enabling to choose between them.

Originally, FairBatch was designed for binary classification tasks
on data that have a single binary sensitive attribute. Both our
implementations currently stick to that setting, in spite of the
FedFB authors using a formalism that arguably extends formulas to
more generic categorical sensitive attribute(s) - which is not
tested in the paper.

Finally, it is worth noting that the translation of the sampling
probabilities into the data batching process is done in accordance
with the reference implementation by the original FairBatch authors.
More details may be found in the documentation of `FairbatchDataset`
(a backend tool that end-users do not need to use directly).

Controllers
-----------
* [FairbatchControllerClient]
[declearn.fairness.fairbatch.FairbatchControllerClient]:
    Client-side controller to implement Fed-FairBatch or FedFB.
* [FairbatchControllerServer]
[declearn.fairness.fairbatch.FairbatchControllerServer]:
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

References
----------
- [1]
    Roh et al. (2020).
    FairBatch: Batch Selection for Model Fairness.
    https://arxiv.org/abs/2012.01696
- [2]
    Zeng et al. (2022).
    Improving Fairness via Federated Learning.
    https://arxiv.org/abs/2110.15545
"""

from ._client import FairbatchControllerClient
from ._dataset import FairbatchDataset
from ._fedfb import setup_fedfb_controller
from ._messages import (
    FairbatchOkay,
    FairbatchSamplingProbas,
)
from ._sampling import (
    FairbatchSamplingController,
    setup_fairbatch_controller,
)
from ._server import FairbatchControllerServer
