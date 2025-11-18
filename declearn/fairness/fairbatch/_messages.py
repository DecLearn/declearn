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

"""Fed-FairBatch/Fed-FB specific messages."""

import dataclasses
from typing import List

from declearn.messaging import Message

__all__ = [
    "FairbatchOkay",
    "FairbatchSamplingProbas",
]


@dataclasses.dataclass
class FairbatchOkay(Message):
    """Message for client signal that Fed-FairBatch/FedFB update went fine."""

    typekey = "fairbatch-okay"


@dataclasses.dataclass
class FairbatchSamplingProbas(Message):
    """Message for server-emitted Fed-FairBatch/Fed-FB sampling probabilities.

    Fields
    ------
    probas:
        List of group-wise sampling probabilities, ordered based on
        an agreed-upon sorted list of sensitive groups.
    """

    probas: List[float]

    typekey = "fairbatch-probas"
