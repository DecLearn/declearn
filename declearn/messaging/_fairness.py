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

"""Messages for fairness-aware federated learning setup and rounds."""

import dataclasses
from typing import Any, Dict, List, Optional, Self, Tuple

from declearn.messaging._api import Message
from declearn.model.api import Vector

__all__ = [
    "FairnessCounts",
    "FairnessGroups",
    "FairnessQuery",
    "FairnessReply",
    "FairnessSetupQuery",
]


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
class FairnessQuery(Message):
    """Base Message for server-emitted fairness-computation queries.

    This message conveys hyper-parameters used when evaluating a model's
    accuracy and/or loss over group-wise samples (from which fairness is
    derived). Model weights may be attached.

    Algorithm-specific information should be conveyed using ad-hoc
    messages exchanged as part of fairness-enforcement routines.
    """

    typekey = "fairness-request"

    round_i: int
    batch_size: int = 32
    n_batch: Optional[int] = None
    thresh: Optional[float] = None
    weights: Optional[Vector] = None


@dataclasses.dataclass
class FairnessReply(Message):
    """Base Message for client-emitted fairness-computation results.

    This message conveys results from the evaluation of a model's accuracy
    and/or loss over group-wise samples (from which fairness is derived).

    This information is generically stored as a list of `values`, the
    mearning and structure of which is left up to algorithm-specific
    controllers.
    """

    typekey = "fairness-reply"

    values: List[float] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class FairnessSetupQuery(Message):
    """Message to instruct clients to instantiate a fairness controller.

    Fields
    ------
    algorithm:
        Name of the algorithm, under which the target controller type
        is expected to be registered.
    params:
        Dict of instantiation keyword arguments to the controller.
    """

    typekey = "fairness-setup-query"

    algorithm: str
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)
