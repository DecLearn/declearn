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

"""Utils for SecAgg features and schemes.

Prime number generation utils
-----------------------------

* [generate_random_biprime][declearn.secagg.utils.generate_random_biprime]:
    Generate a random biprime integer with a target bit length.
* [generate_random_prime][declearn.secagg.utils.generate_random_prime]:
    Generate a random prime integer with given bit length.

Public-Key Infrastructure utils
-------------------------------

* [IdentityKeys][declearn.secagg.utils.IdentityKeys]:
    Handler to hold and load long-lived Ed25519 identity keys.

Quantization utils
------------------

* [Quantizer][declearn.secagg.utils.Quantizer]:
    Data (un)quantization facility for finite-domain int/float conversion.

Miscellaneous utils
-------------------

* [get_numpy_float_dtype][declearn.secagg.utils.get_numpy_float_dtype]:
    Return the smallest-size numpy float dtype for a given values range.
* [get_numpy_uint_dtype][declearn.secagg.utils.get_numpy_uint_dtype]:
    Return the smallest-size numpy uint dtype for a given integer range.
"""

from ._ed25519 import IdentityKeys
from ._numpy import (
    get_numpy_float_dtype,
    get_numpy_uint_dtype,
)
from ._prime import (
    generate_random_biprime,
    generate_random_prime,
)
from ._quantize import Quantizer
