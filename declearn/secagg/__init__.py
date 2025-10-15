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

"""Secure Aggregation API, methods and utils.

This submodule implements Secure Aggregation (SecAgg), that is the
possibility to encrypt client-wise shared values and decrypt their
aggregate over all clients without revealing initial values to the
server (nor other clients).

It provides both an ensemble of API-defining abstractions and some
practical implementations of methods taken from the litterature.

The abstractions are two-fold:

- on the one hand, controllers for the encryption, aggregation and
  decryption of values are written in a rather generic way (albeit
  interfacing some DecLearn-specific types in addition to standard
  ones);
- on the other hand, setup routines to instantiate such controllers
  during a process rely on the DecLearn network communication API.


Core submodules
---------------

These submodules expose API-defining abstractions and backbone utils
to implement SecAgg and integrate it into other parts of DecLearn.

* [api][declearn.secagg.api]:
    API-defining ABCs for SecAgg controllers and encrypted data.
* [messaging][declearn.secagg.messaging]:
    SecAgg counterparts to some default Federated Learning messages.
* [utils][declearn.secagg.utils]:
    Utils for SecAgg features and schemes.


SecAgg types
------------

These submodules hold concrete implementations of specific SecAgg methods.

* [joye_libert][declearn.secagg.joye_libert]:
    Secure Aggregation tools based on Joye-Libert homomorphic summation.
* [masking][declearn.secagg.masking]:
    Secure Aggregation tools based on values' masking with shared RNG seeds.


Security protocols
------------------

These protocols do not implement SecAgg per se, but are instrumental
in the setup of one or more SecAgg types.

* [shamir][declearn.secagg.shamir]:
    Shamir secret-sharing tools.
* [x3dh][declearn.secagg.x3dh]:
    Extended Triple Diffie-Hellman (X3DH) key agreement tools.


Utility functions
-----------------
* [list_available_secagg_types][declearn.secagg.list_available_secagg_types]:
    List available SecAgg types and access associated config types.
* [parse_secagg_config_client][declearn.secagg.parse_secagg_config_client]:
    Parse input arguments into a `SecaggConfigClient` instance.
* [parse_secagg_config_server][declearn.secagg.parse_secagg_config_server]:
    Parse input arguments into a `SecaggConfigServer` instance.
"""

from . import api, joye_libert, masking, messaging, shamir, utils, x3dh
from ._setup import (
    list_available_secagg_types,
    parse_secagg_config_client,
    parse_secagg_config_server,
)
