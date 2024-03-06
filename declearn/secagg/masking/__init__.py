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

"""Masked summation tools.

This module implements controllers to conduct secure aggregation of
values by generating and incorporating pseudo-random masks into the
shared values that cancel out via summation.

These controllers require peers to have previously agreed on pairwise
RNG seeds from which masks are derived.

Controllers
-----------

* [MaskingDecrypter][declearn.secagg.masking.MaskingDecrypter]:
    Controller for the reconstruction of sums of masked values.
* [MaskingEncrypter][declearn.secagg.masking.MaskingEncrypter]:
    Controller for the masking of values that need summation.


Aggregate
---------

* [MaskedAggregate][declearn.secagg.masking.MaskedAggregate]:
    'Aggregate'-like container for masked quantized values
"""

from ._aggregate import MaskedAggregate
from ._encrypt import MaskingEncrypter
from ._decrypt import MaskingDecrypter
