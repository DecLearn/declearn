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

Note that apart from the possible loss of information due to quantization
on an integer field, this masking-based SecAgg scheme has a very limited
overhead of computation and communication costs, especially when a limited
quantization bitsize B (typically, 64) is set, as encrypted values will in
that case be represented as `uint64` values.

Controllers
-----------

* [MaskingDecrypter][declearn.secagg.masking.MaskingDecrypter]:
    Controller for the reconstruction of sums of masked values.
* [MaskingEncrypter][declearn.secagg.masking.MaskingEncrypter]:
    Controller for the masking of values that need summation.

Setup & config
--------------

* [MaskingSecaggConfigClient]\
[declearn.secagg.masking.MaskingSecaggConfigClient]:
    Client-side config and setup controller for masking-based SecAgg.
* [MaskingSecaggConfigServer]\
[declearn.secagg.masking.MaskingSecaggConfigServer]:
    Server-side config and setup controller for masking-based SecAgg.
* [messages][declearn.secagg.masking.messages]:
    Submodule providing with messages for masking-based SecAgg setup routines.

Aggregate
---------

* [MaskedAggregate][declearn.secagg.masking.MaskedAggregate]:
    'Aggregate'-like container for masked quantized values

References
----------
- [1]
    Bonawitz et al., 2016.
    Practical Secure Aggregation for Federated Learning
    on User-Held Data.
    https://arxiv.org/abs/1611.04482
- [2]
    Marlinspike & Perrin, 2016.
    The X3DH Key Agreement Protocol.
    https://www.signal.org/docs/specifications/x3dh/
"""

from . import messages
from ._aggregate import MaskedAggregate
from ._decrypt import MaskingDecrypter
from ._encrypt import MaskingEncrypter
from ._setup import (
    MaskingSecaggConfigClient,
    MaskingSecaggConfigServer,
)
