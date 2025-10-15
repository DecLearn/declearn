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

"""API and default classes to define parsable messages for applications.

Message API and tools
---------------------

* [Message][declearn.messaging.Message]:
    Abstract base dataclass to define parsable messages.
* [SerializedMessage][declearn.messaging.SerializedMessage]:
    Container for serialized Message instances.


Base messages
-------------

* [CancelTraining][declearn.messaging.CancelTraining]
* [Error][declearn.messaging.Error]
* [EvaluationReply][declearn.messaging.EvaluationReply]
* [EvaluationRequest][declearn.messaging.EvaluationRequest]
* [GenericMessage][declearn.messaging.GenericMessage]
* [InitRequest][declearn.messaging.InitRequest]
* [InitReply][declearn.messaging.InitReply]
* [MetadataQuery][declearn.messaging.MetadataQuery]
* [MetadataReply][declearn.messaging.MetadataReply]
* [PrivacyRequest][declearn.messaging.PrivacyRequest]
* [PrivacyReply][declearn.messaging.PrivacyReply]
* [StopTraining][declearn.messaging.StopTraining]
* [TrainReply][declearn.messaging.TrainReply]
* [TrainRequest][declearn.messaging.TrainRequest]

Fairness algorithms messages
----------------------------

* [FairnessCounts][declearn.messaging.FairnessCounts]
* [FairnessGroups][declearn.messaging.FairnessGroups]
* [FairnessQuery][declearn.messaging.FairnessQuery]
* [FairnessReply][declearn.messaging.FairnessReply]
* [FairnessSetupQuery][declearn.messaging.FairnessSetupQuery]
"""

from ._api import (
    Message,
    SerializedMessage,
)
from ._base import (
    CancelTraining,
    Error,
    EvaluationReply,
    EvaluationRequest,
    GenericMessage,
    InitReply,
    InitRequest,
    MetadataQuery,
    MetadataReply,
    PrivacyReply,
    PrivacyRequest,
    StopTraining,
    TrainReply,
    TrainRequest,
)
from ._fairness import (
    FairnessCounts,
    FairnessGroups,
    FairnessQuery,
    FairnessReply,
    FairnessSetupQuery,
)
