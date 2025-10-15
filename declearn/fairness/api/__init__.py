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

"""Abstract and base components for fairness-aware federated learning.

Endpoint Controller ABCs
------------------------

* [FairnessControllerClient][declearn.fairness.api.FairnessControllerClient]:
    Abstract base class for client-side fairness controllers.
* [FairnessControllerServer][declearn.fairness.api.FairnessControllerServer]:
    Abstract base class for server-side fairness controllers.

Group-fairness functions
------------------------
API-defining ABC and generic constructor:

* [FairnessFunction][declearn.fairness.api.FairnessFunction]:
    Abstract base class for group-fairness functions.
* [instantiate_fairness_function]\
[declearn.fairness.api.instantiate_fairness_function]:
    Instantiate a FairnessFunction from its specifications.

Built-in concrete implementations may be found in [declearn.fairness.core][].

Dataset subclass
----------------

* [FairnessDataset][declearn.fairness.api.FairnessDataset]:
    Abstract base class for Fairness-aware `Dataset` interfaces.

Backend
-------

* [FairnessMetricsComputer][declearn.fairness.api.FairnessMetricsComputer]:
    Utility dataset-handler to compute group-wise evaluation metrics.
"""

from ._client import FairnessControllerClient
from ._dataset import FairnessDataset
from ._fair_func import FairnessFunction, instantiate_fairness_function
from ._metrics import FairnessMetricsComputer
from ._server import FairnessControllerServer
