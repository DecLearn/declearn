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

"""Secure Aggregation features and utils.

* [joye_libert][declearn.secagg.joye_libert]:
    Joye-Libert homomorphic summation tools.
* [shamir][declearn.secagg.shamir]:
    Shamir secret-sharing tools.
* [x3dh][declearn.secagg.x3dh]:
    Extended Triple Diffie-Hellman (X3DH) key agreement tools.
* [utils][declearn.secagg.utils]:
    Utils for SecAgg features and schemes.
"""

from . import joye_libert
from . import shamir
from . import utils
from . import x3dh
