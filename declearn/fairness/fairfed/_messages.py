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

"""FairFed specific messages."""

import dataclasses
from typing import Self

from declearn.messaging import Message
from declearn.secagg.api import Decrypter, Encrypter
from declearn.secagg.messaging import SecaggMessage

__all__ = [
    "FairfedDelta",
    "FairfedDeltavg",
    "FairfedFairness",
    "FairfedOkay",
    "SecaggFairfedDelta",
]


@dataclasses.dataclass
class FairfedOkay(Message):
    """Message for client-emitted signal that Fairfed update went fine."""

    typekey = "fairfed-okay"


@dataclasses.dataclass
class FairfedFairness(Message):
    """Message for server-emitted Fairfed global fairness value sharing.

    Fields
    ------
    fairness:
        Global fairness (or accuracy) value.
    """

    fairness: float

    typekey = "fairfed-fairness"


@dataclasses.dataclass
class FairfedDelta(Message):
    """Message for client-emitted Fairfed absolute fairness difference.

    Fields
    ------
    delta:
        Local absolute difference in fairness (or accuracy).
    """

    delta: float

    typekey = "fairfed-delta"


@dataclasses.dataclass
class SecaggFairfedDelta(SecaggMessage[FairfedDelta]):
    """SecAgg-wrapped 'FairfedDelta' message."""

    typekey = "secagg-fairfed-delta"

    delta: int

    @classmethod
    def from_cleartext_message(
        cls,
        cleartext: FairfedDelta,
        encrypter: Encrypter,
    ) -> Self:
        delta = encrypter.encrypt_float(cleartext.delta)
        return cls(delta=delta)

    def decrypt_wrapped_message(
        self,
        decrypter: Decrypter,
    ) -> FairfedDelta:
        delta = decrypter.decrypt_float(self.delta)
        return FairfedDelta(delta=delta)

    def aggregate(
        self,
        other: Self,
        decrypter: Decrypter,
    ) -> Self:
        delta = decrypter.sum_encrypted([self.delta, other.delta])
        return self.__class__(delta=delta)


@dataclasses.dataclass
class FairfedDeltavg(Message):
    """Message for server-emitted Fairfed average absolute fairness difference.

    Fields
    ------
    deltavg:
        Average absolute difference in fairness (or accuracy).
    """

    deltavg: float

    typekey = "fairfed-deltavg"
