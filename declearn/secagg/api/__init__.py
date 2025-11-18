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

"""API-defining ABCs for SecAgg controllers and encrypted data.

ABCs for SecAgg usage
---------------------

* [Decrypter][declearn.secagg.api.Decrypter]:
    ABC controller for the decryption of summed encrypted values
* [Encrypter][declearn.secagg.api.Encrypter]:
    ABC controller for the encryption of values that need summation.
* [SecureAggregate][declearn.secagg.api.SecureAggregate]:
    Abstract 'Aggregate'-like wrapper for encrypted 'Aggregate' objects.

ABCs for SecAgg setup
---------------------
* [SecaggConfigClient][declearn.secagg.api.SecaggConfigClient]:
    ABC for client-side SecAgg configuration and setup.
* [SecaggConfigServer][declearn.secagg.api.SecaggConfigServer]:
    ABC for server-side SecAgg configuration and setup.
* [SecaggSetupQuery][declearn.secagg.api.SecaggSetupQuery]:
    ABC message for all SecAgg setup init requests


Type-hint aliases
-----------------
* [ArraySpec][declearn.secagg.api.ArraySpec]:
    Specification of a numpy array's dtype and shape.
* [EncryptedSpecs][declearn.secagg.api.EncryptedSpecs]:
    Specification of flattened encrypted Aggregate fields.
"""

from ._aggregate import ArraySpec, EncryptedSpecs, SecureAggregate
from ._decrypt import Decrypter
from ._encrypt import Encrypter
from ._setup import (
    SecaggConfigClient,
    SecaggConfigServer,
    SecaggSetupQuery,
)
