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

"""Messages for X3DH setup routines."""

import dataclasses
from typing import List

from declearn.messaging import Message

__all__ = [
    "X3DHOkay",
    "X3DHRequests",
    "X3DHResponses",
    "X3DHTrigger",
]


@dataclasses.dataclass
class X3DHTrigger(Message):
    """Message to instruct a peer to generate X3DH requests."""

    typekey = "x3dh-trigger"

    n_reqs: int


@dataclasses.dataclass
class X3DHRequests(Message):
    """Message to transmit a list of X3DH setup requests."""

    typekey = "x3dh-requests"

    requests: List[int]


@dataclasses.dataclass
class X3DHResponses(Message):
    """Message to transmit a list of X3DH setup responses."""

    typekey = "x3dh-responses"

    responses: List[int]


@dataclasses.dataclass
class X3DHOkay(Message):
    """Message to signal that X3DH setup was properly finalized."""

    typekey = "x3dh-okay"
