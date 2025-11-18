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

"""Processes and components for fairness-aware federated learning.

Introduction
------------

This modules provides with a general API and some specific algorithms
to measure and enforce group fairness as part of a federated learning
process in DecLearn.

Group fairness refers to a setting where a classifier is trained over
data that can be split between various subsets based on one or more
categorical sensitive attributes, usually comprising the target label.
In such a setting, the model's fairness is defined and evaluated by
comparing its accuracy over the various subgroups, using one of the
various definitions proposed in the litterature.

The algorithms and shared API implemented in this module consider that
the fairness being measured (and optimized) is to be computed over the
union of all training datasets held by clients. The API is designed to
be compatible with any number of sensitive groups, with regimes where
individual clients do not necessarily hold samples to each and every
group, and with all group fairness definitions that can be expressed
in a form that was introduced in paper [1]. However, some restrictions
may be enforced by concrete algorithms, in alignment with those set by
their original authors.

Currently, concrete algorithms include:

- Fed-FairGrad, adapted from [1]
- Fed-FairBatch, adapted from [2], and the FedFB variant based on [3]
- FairFed, based on [4]

In addition, a "monitor-only" algorithm is provided, that merely uses
the shared API to measure client-wise and global fairness throughout
training without altering the training algorithm.


API-defining and core submodules
--------------------------------

* [api][declearn.fairness.api]:
    Abstract and base components for fairness-aware federated learning.
* [core][declearn.fairness.core]:
    Built-in concrete components for fairness-aware federated learning.

Algorithms submodules
---------------------

* [fairbatch][declearn.fairness.fairbatch]:
    Fed-FairBatch / FedB algorithm controllers and utils.
* [fairfed][declearn.fairness.fairfed]:
    FairFed algorithm controllers and utils.
* [fairgrad][declearn.fairness.fairgrad]:
    Fed-FairGrad algorithm controllers and utils.
* [monitor][declearn.fairness.monitor]:
    Fairness-monitoring controllers, that leave training unaltered.

Note that the controllers implemented under these submodules
are type-registered under the submodule's name.

References
----------

- [1]
    Maheshwari & Perrot (2023).
    FairGrad: Fairness Aware Gradient Descent.
    https://openreview.net/forum?id=0f8tU3QwWD
- [2]
    Roh et al. (2020).
    FairBatch: Batch Selection for Model Fairness.
    https://arxiv.org/abs/2012.01696
- [3]
    Zeng et al. (2022).
    Improving Fairness via Federated Learning.
    https://arxiv.org/abs/2110.15545
- [4]
    Ezzeldin et al. (2021).
    FairFed: Enabling Group Fairness in Federated Learning
    https://arxiv.org/abs/2110.00857
"""

from . import api, core, fairbatch, fairfed, fairgrad, monitor
