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

"""Utils to fetch and prepare some open-source datasets.

Datasets
--------
* [load_heart_uci][declearn.dataset.examples.load_heart_uci]:
    Load and/or download a pre-processed UCI heart disease dataset.
* [load_mnist][declearn.dataset.examples.load_mnist]:
    Load and/or download the MNIST digit-classification dataset.
"""

from ._heart_uci import load_heart_uci
from ._mnist import load_mnist
