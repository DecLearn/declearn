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

"""Messages for Joye-Libert SecAgg setup routines."""

import dataclasses
from typing import Dict

from declearn.messaging import Message
from declearn.secagg.api import SecaggSetupQuery

__all__ = [
    "JoyeLibertPeerInfo",
    "JoyeLibertPublicShare",
    "JoyeLibertSecaggSetupQuery",
    "JoyeLibertSecretShares",
    "JoyeLibertShamirPrime",
]


@dataclasses.dataclass
class JoyeLibertSecaggSetupQuery(SecaggSetupQuery):
    """Message to trigger Joye-Libert SecAgg setup initialization."""

    typekey = "jls-setup-query"

    bitsize: int
    clipval: float


@dataclasses.dataclass
class JoyeLibertPeerInfo(Message):
    """Message to share a peer's public information for Joye-Libert setup."""

    typekey = "jls-peer-info"

    biprime: int
    id_key: str


@dataclasses.dataclass
class JoyeLibertShamirPrime(Message):
    """Message to transmit a public large prime for Shamir Secret Sharing."""

    typekey = "jls-shamir-prime"

    prime: int


@dataclasses.dataclass
class JoyeLibertSecretShares(Message):
    """Message to transmit encrypted shares of Joye-Libert secret keys."""

    typekey = "jls-secret-shares"

    shares: Dict[str, str]


@dataclasses.dataclass
class JoyeLibertPublicShare(Message):
    """Message to transmit a public share to a Joye-Libert public key."""

    typekey = "jls-public-share"

    share: int
