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

This module implements primitives and controllers to conduct secure
aggregation of values using a homomorphic summation algorithm from
Marc Joye & Benoît Libert published in 2013 [1].

Primitives
----------

* [DEFAULT_BIPRIME][declearn.secagg.joye_libert.DEFAULT_BIPRIME]:
    Default Biprime value used as modulus in Joye-Libert functions.
* [encrypt][declearn.secagg.joye_libert.encrypt]:
    Apply Joye-Libert encryption to an integer value.
* [decrypt_sum][declearn.secagg.joye_libert.decrypt_sum]:
    Apply Joye-Libert decryption to an encrypted sum of private values.
 * [sum_encrypted][declearn.secagg.joye_libert.sum_encrypted]:
    Apply homomorphic summation to some Joye-Libert encrypted values.

References
----------
[1] Joye & Libert, 2013.
    A Scalable Scheme for Privacy-Preserving Aggregation
    of Time-Series Data.
    https://marcjoye.github.io/papers/JL13aggreg.pdf
"""

from ._joye_libert import (
    DEFAULT_BIPRIME,
    encrypt,
    decrypt_sum,
    sum_encrypted,
)
