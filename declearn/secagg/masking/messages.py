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

"""Messages for Masking-based SecAgg setup routines."""

import dataclasses

from declearn.messaging import Message
from declearn.secagg.api import SecaggSetupQuery

__all__ = [
    "MaskingSecaggSetupQuery",
    "MaskingSecaggSetupReply",
]


@dataclasses.dataclass
class MaskingSecaggSetupQuery(SecaggSetupQuery):
    """Server-emitted message to trigger masking-based SecAgg setup."""

    typekey = "masking-secagg-setup-query"


@dataclasses.dataclass
class MaskingSecaggSetupReply(Message):
    """Client-emitted empty message signaling that SecAgg setup is fine."""

    typekey = "masking-secagg-setup-reply"
