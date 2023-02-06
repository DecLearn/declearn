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

"""Optimizer loss-regularization algorithms, implemented as plug-in modules.

Base class implemented here:
* Regularizer: base API for loss-regularization plug-ins

Common regularization terms:
* LassoRegularizer : L1 regularization, aka Lasso penalization
* RidgeRegularizer : L2 regularization, aka Ridge penalization

Federated-Learning-specific regularizers:
* FedProxRegularizer : FedProx algorithm, as a proximal term regularizer
"""

from ._api import Regularizer
from ._base import (
    FedProxRegularizer,
    LassoRegularizer,
    RidgeRegularizer,
)
