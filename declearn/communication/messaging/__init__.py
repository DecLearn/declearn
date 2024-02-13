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

"""DEPRECATED submodule defining messaging containers and flags.

This submodule was deprecated in DecLearn 2.4 in favor of `declearn.messaging`.
It should no longer be used, and will be removed in 2.6 and/or 3.0.

Most of its contents are re-exports of non-deprecated classes and functions
from 'declearn.messaging'. Others will trigger deprecation warnings (and may
cause failures) if used.

Deprecated classes uniquely-defined here are:

* [Empty][declearn.communication.messaging.Empty]
* [GetMessageRequest][declearn.communication.messaging.GetMessageRequest]
* [JoinReply][declearn.communication.messaging.JoinReply]
* [JoinRequest][declearn.communication.messaging.JoinRequest]

The `flags` submodule is also re-exported, but should preferably be imported
as `declearn.communication.api.backend.flags`.
"""

from declearn.communication.api.backend import flags

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
