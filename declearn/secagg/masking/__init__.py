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

"""Secure Aggregation tools based on values' masking with shared RNG seeds.

This module implements controllers to conduct secure aggregation of
values by generating and incorporating pseudo-random masks into the
shared values that cancel out via summation.

These controllers require peers to have previously agreed on pairwise
RNG seeds from which masks are derived. Both controllers and proposed
setup routines are loosely based on the protocols published by Bonawitz
et al. in 2016 [1]. The most salient differences are that (a) we leave
thresholding apart, (b) we use the X3DH protocol [2] to set up pairwise
secrets, (c) we require a pre-existing public key infrastructure rather
than put trust in the server to bootstrap and distribute identity keys.

Controllers
-----------

* [MaskingDecrypter][declearn.secagg.masking.MaskingDecrypter]:
    Controller for the reconstruction of sums of masked values.
* [MaskingEncrypter][declearn.secagg.masking.MaskingEncrypter]:
    Controller for the masking of values that need summation.


Setup routines
--------------

* [run_masking_secagg_setup_client]\
[declearn.secagg.masking.run_masking_secagg_setup_client]:
    Participate in a masking-based SecAgg setup protocol.
* [run_masking_secagg_setup_server]\
[declearn.secagg.masking.run_masking_secagg_setup_server]:
    Orchestrate a masking-based SecAgg setup protocol.
* [messages][declearn.secagg.masking.messages]:
    Submodule providing with messages for masking-based SecAgg setup routines.

Aggregate
---------

* [MaskedAggregate][declearn.secagg.masking.MaskedAggregate]:
    'Aggregate'-like container for masked quantized values

References
----------
[1] Bonawitz et al., 2016.
    Practical Secure Aggregation for Federated Learning
    on User-Held Data.
    https://arxiv.org/abs/1611.04482
[2] Marlinspike & Perrin, 2016.
    The X3DH Key Agreement Protocol.
    https://www.signal.org/docs/specifications/x3dh/
"""

from ._aggregate import MaskedAggregate
from ._encrypt import MaskingEncrypter
from ._decrypt import MaskingDecrypter
