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

"""Processes and components for fairness-aware federated learning.

This module implements the following submodules:

* [api][declearn.fairness.api]:
    API to set up and run fairness-aware federated learning algorithms.
* [core][declearn.fairness.core]:
    Core components and utils for fairness-aware (federated) machine learning.
* [fairbatch][declearn.fairness.fairbatch]:
    Fed-FairBatch / FedB algorithm controllers and utils.
* [fairfed][declearn.fairness.fairfed]:
    FairFed algorithm controllers and utils.
* [fairgrad][declearn.fairness.fairgrad]:
    Fed-FairGrad algorithm controllers and utils.
"""

from . import core
from . import api
from . import fairbatch
from . import fairfed
from . import fairgrad
