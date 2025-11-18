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

"""Fed-FairGrad specific messages."""

import dataclasses
from typing import List

from declearn.messaging import Message

__all__ = [
    "FairgradOkay",
    "FairgradWeights",
]


@dataclasses.dataclass
class FairgradOkay(Message):
    """Message for client-emitted signal that Fed-FairGrad update went fine."""

    typekey = "fairgrad-okay"


@dataclasses.dataclass
class FairgradWeights(Message):
    """Message for server-emitted (Fed-)FairGrad loss weights sharing.

    Fields
    ------
    weights:
        List of group-wise loss weights, ordered based on
        an agreed-upon sorted list of sensitive groups.
    """

    weights: List[float]

    typekey = "fairgrad-weights"
