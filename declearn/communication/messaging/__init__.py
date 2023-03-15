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

"""Submodule defining messaging containers and flags used in declearn.

The submodule exposes the [Message][declearn.communication.messaging.Message]
abstract base dataclass, and its children:

* [CancelTraining][declearn.communication.messaging.CancelTraining]
* [Empty][declearn.communication.messaging.Empty]
* [Error][declearn.communication.messaging.Error]
* [GenericMessage][declearn.communication.messaging.GenericMessage]
* [GetMessageRequest][declearn.communication.messaging.GetMessageRequest]
* [EvaluationReply][declearn.communication.messaging.EvaluationReply]
* [EvaluationRequest][declearn.communication.messaging.EvaluationRequest]
* [InitRequest][declearn.communication.messaging.InitRequest]
* [JoinReply][declearn.communication.messaging.JoinReply]
* [JoinRequest][declearn.communication.messaging.JoinRequest]
* [PrivacyRequest][declearn.communication.messaging.PrivacyRequest]
* [StopTraining][declearn.communication.messaging.StopTraining]
* [TrainReply][declearn.communication.messaging.TrainReply]
* [TrainRequest][declearn.communication.messaging.TrainRequest]

It also exposes the [parse_message_from_string]\
[declearn.communication.messaging.parse_message_from_string]
function to recover the structures above from a dump string.

Finally, it exposes a set of [flags][declearn.communication.messaging.flags],
as constant strings that may be used conventionally as part of messages.
"""

from . import flags
from ._messages import (
    CancelTraining,
    Empty,
    Error,
    GenericMessage,
    GetMessageRequest,
    EvaluationReply,
    EvaluationRequest,
    InitRequest,
    JoinReply,
    JoinRequest,
    Message,
    PrivacyRequest,
    StopTraining,
    TrainReply,
    TrainRequest,
    parse_message_from_string,
)
