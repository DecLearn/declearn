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

"""Secure Aggregation tools based on Joye-Libert homomorphic summation.

This module implements primitives and controllers to conduct secure
aggregation of values using a homomorphic summation algorithm from
Marc Joye & Benoît Libert published in 2013 [1].

Note that Joye-Libert encryption requires operating over large integer
fields, resulting in a relatively-costly SecAgg scheme, both in terms
of computations and communications, as usual 32-64 bit values end up
converted to (possibly very) large integers.


Controllers
-----------

* [JoyeLibertDecrypter][declearn.secagg.joye_libert.JoyeLibertDecrypter]:
    Controller for the decryption of (homomorphic) sums of encrypted values.
* [JoyeLibertEncrypter][declearn.secagg.joye_libert.JoyeLibertEncrypter]:
    Controller for the encryption of values that need homomorphic summation.

Setup & config
--------------

* [JoyeLibertSecaggConfigClient]\
[declearn.secagg.joye_libert.JoyeLibertSecaggConfigClient]:
    Client-side config and setup controller for Joye-Libert SecAgg.
* [JoyeLibertSecaggConfigServer]\
[declearn.secagg.joye_libert.JoyeLibertSecaggConfigServer]:
    Server-side config and setup controller for Joye-Libert SecAgg.
* [messages][declearn.secagg.joye_libert.messages]:
    Submodule providing with messages for Joye-Libert SecAgg setup routines.

Primitives
----------

* [DEFAULT_BIPRIME][declearn.secagg.joye_libert.DEFAULT_BIPRIME]:
    Default Biprime value used as modulus in Joye-Libert functions.
* [decrypt_sum][declearn.secagg.joye_libert.decrypt_sum]:
    Apply Joye-Libert decryption to an encrypted sum of private values.
* [encrypt][declearn.secagg.joye_libert.encrypt]:
    Apply Joye-Libert encryption to an integer value.
 * [sum_encrypted][declearn.secagg.joye_libert.sum_encrypted]:
    Apply homomorphic summation to some Joye-Libert encrypted values.

Aggregate
---------

* [JLSAggregate][declearn.secagg.joye_libert.JLSAggregate]:
    'Aggregate'-like container for Joye-Libert encrypted values.

References
----------
[1] Joye & Libert, 2013.
    A Scalable Scheme for Privacy-Preserving Aggregation
    of Time-Series Data.
    https://marcjoye.github.io/papers/JL13aggreg.pdf
"""

from . import messages
from ._aggregate import JLSAggregate
from ._decrypt import JoyeLibertDecrypter
from ._encrypt import JoyeLibertEncrypter
from ._primitives import (
    DEFAULT_BIPRIME,
    decrypt_sum,
    encrypt,
    sum_encrypted,
)
from ._setup import (
    JoyeLibertSecaggConfigClient,
    JoyeLibertSecaggConfigServer,
)
