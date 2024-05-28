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

"""API messages for fairness-aware federated learning setup and rounds."""

import dataclasses
from typing import Any, List, Optional, Tuple

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.messaging import Message
from declearn.secagg.api import Decrypter, Encrypter
from declearn.secagg.messaging import SecaggMessage

__all__ = [
    "FairnessAccuracy",
    "FairnessCounts",
    "FairnessGroups",
    "FairnessRoundQuery",
    "FairnessRoundReply",
    "SecaggFairnessAccuracy",
    "SecaggFairnessCounts",
]


@dataclasses.dataclass
class FairnessAccuracy(Message):
    """Message for client-emitted model accuracy across sensitive groups.

    Fields
    ------
    values:
        List of group-wise accuracy values, ordered based
        on an agreed-upon sorted list of sensitive groups.
    """

    values: List[float]

    typekey = "fairness-accuracy"


@dataclasses.dataclass
class SecaggFairnessAccuracy(SecaggMessage[FairnessAccuracy]):
    """SecAgg counterpart of the 'FairnessAccuracy' message class."""

    values: List[int]

    typekey = "secagg-fairness-accuracy"

    @classmethod
    def from_cleartext_message(
        cls,
        cleartext: FairnessAccuracy,
        encrypter: Encrypter,
    ) -> Self:
        values = [encrypter.encrypt_float(val) for val in cleartext.values]
        return cls(values=values)

    def decrypt_wrapped_message(
        self,
        decrypter: Decrypter,
    ) -> FairnessAccuracy:
        values = [decrypter.decrypt_float(val) for val in self.values]
        return FairnessAccuracy(values=values)

    def aggregate(
        self,
        other: Self,
        decrypter: Decrypter,
    ) -> Self:
        values = [
            decrypter.sum_encrypted([v_a, v_b])
            for v_a, v_b in zip(self.values, other.values)
        ]
        return self.__class__(values=values)


@dataclasses.dataclass
class FairnessCounts(Message):
    """Message for client-emitted sample counts across sensitive groups.

    Fields
    ------
    counts:
        List of group-wise sample counts, ordered based on
        an agreed-upon sorted list of sensitive groups.
    """

    counts: List[int]

    typekey = "fairness-counts"


@dataclasses.dataclass
class SecaggFairnessCounts(SecaggMessage[FairnessCounts]):
    """SecAgg counterpart of the 'FairnessCounts' message class."""

    counts: List[int]

    typekey = "secagg-fairness-counts"

    @classmethod
    def from_cleartext_message(
        cls,
        cleartext: FairnessCounts,
        encrypter: Encrypter,
    ) -> Self:
        counts = [encrypter.encrypt_uint(val) for val in cleartext.counts]
        return cls(counts=counts)

    def decrypt_wrapped_message(
        self,
        decrypter: Decrypter,
    ) -> FairnessCounts:
        counts = [decrypter.decrypt_uint(val) for val in self.counts]
        return FairnessCounts(counts=counts)

    def aggregate(
        self,
        other: Self,
        decrypter: Decrypter,
    ) -> Self:
        counts = [
            decrypter.sum_encrypted([v_a, v_b])
            for v_a, v_b in zip(self.counts, other.counts)
        ]
        return self.__class__(counts=counts)


@dataclasses.dataclass
class FairnessGroups(Message):
    """Message to exchange a list of unique sensitive group definitions.

    This message may be exchanged both ways, with clients sharing the
    list of groups for which they have samples and the server sharing
    back a unified, sorted list of all sensitive groups across clients.

    Fields
    ------
    groups:
        List of sensitive group definitions, defined by tuples of values
        corresponding to those of one or more sensitive attributes and
        (optionally) a target label.
    """

    groups: List[Tuple[Any, ...]]

    typekey = "fairness-groups"

    @classmethod
    def from_kwargs(
        cls,
        **kwargs: Any,
    ) -> Self:
        kwargs["groups"] = [tuple(group) for group in kwargs["groups"]]
        return super().from_kwargs(**kwargs)


@dataclasses.dataclass
class FairnessRoundQuery(Message):
    """Base Message for server-emitted fairness-computation queries.

    The base `FairnessRoundQuery` defines information that is used
    when evaluating a model's accuracy and/or loss over group-wise
    training samples.

    Subclasses may be defined to add algorithm-specific information.

    Fields
    ------
    batch_size:
        Number of samples per batch when computing metrics.
    n_batch:
        Optional maximum number of batches to draw per group.
        If None, use the entire wrapped dataset.
    thresh:
        Optional binarization threshold for binary classification
        models' output scores. If None, use 0.5 by default, or 0.0
        for `SklearnSGDModel` instances.
        Unused for multinomial classifiers (argmax over scores).
    """

    batch_size: int = 32
    n_batch: Optional[int] = None
    thresh: Optional[float] = None

    typekey = "fairness-round-query"


@dataclasses.dataclass
class FairnessRoundReply(Message):
    """Base Message for client-emitted fairness-round end signal.

    By default this message is empty, merely noticing that things
    went well. Subclasses may be used to convey algorithm-specific
    results or information.
    """

    typekey = "fairness-round-reply"
