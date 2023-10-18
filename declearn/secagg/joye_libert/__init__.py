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

"""Joye-Libert homomorphic summation tools.

* [DEFAULT_BIPRIME][declearn.secagg.joye_libert.DEFAULT_BIPRIME]:
    Default Biprime value used as modulus in Joye-Libert functions.
* [encrypt][declearn.secagg.joye_libert.encrypt]:
    Apply Joye-Libert encryption to an integer value.
* [sum_decrypt][declearn.secagg.joye_libert.sum_decrypt]:
    Apply Joye-Libert aggregate decryption of a list of encrypted integers.
"""

from ._joye_libert import (
    DEFAULT_BIPRIME,
    encrypt,
    sum_decrypt,
)
