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

"""Model training and evaluation orchestration tools.

Classes
-------
The main class implemented here is `TrainingManager`, that is used by clients
and may also be used to perform centralized machine learning using declearn:

* [TrainingManager][declearn.training.TrainingManager]:
    Class wrapping the logic for local training and evaluation rounds.

Submodules
----------

* [dp][declearn.training.dp]:
    Differentially-Private training routine utils.
    The main class implemented here is `DPTrainingManager` that implements
    client-side DP-SGD training. This module is to be manually imported or
    lazy-imported (e.g. by `declearn.main.FederatedClient`), and may trigger
    warnings or errors in the absence of the 'opacus' third-party dependency.
"""

from ._manager import TrainingManager
