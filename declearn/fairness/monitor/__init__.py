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

"""Fairness-monitoring controllers, that leave training unaltered.

Introduction
------------
This submodule implements dummy fairness-aware learning controllers, that
implement fairness metrics' computation, hence enabling their monitoring
throughout training, without altering the model's training process itself.

These controllers may therefore be used to monitor fairness metrics of any
baseline federated learning algorithm, notably for comparison purposes with
fairness-aware algorithms implemented using other controllers (FairBatch,
Fed-FairGrad, ...).

Controllers
-----------
* [FairnessMonitorClient][declearn.fairness.monitor.FairnessMonitorClient]:
    Client-side controller to monitor fairness without altering training.
* [FairnessMonitorServer][declearn.fairness.monitor.FairnessMonitorServer]:
    Server-side controller to monitor fairness without altering training.
"""

from ._client import FairnessMonitorClient
from ._server import FairnessMonitorServer
