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

"""Dataclasses defining messages used in declearn communications."""

import abc
import dataclasses
import warnings
from typing import Any, Dict, Optional


from declearn.messaging import (
    CancelTraining,
    Error,
    EvaluationReply,
    EvaluationRequest,
    GenericMessage,
    InitRequest,
    Message,
    PrivacyRequest,
    SerializedMessage,
    StopTraining,
    TrainReply,
    TrainRequest,
)


__all__ = [
    "CancelTraining",
    "Empty",
    "Error",
    "EvaluationReply",
    "EvaluationRequest",
    "GenericMessage",
    "GetMessageRequest",
    "InitRequest",
    "JoinReply",
    "JoinRequest",
    "Message",
    "PrivacyRequest",
    "StopTraining",
    "TrainReply",
    "TrainRequest",
    "parse_message_from_string",
]


@dataclasses.dataclass
class DeprecatedMessage(Message, register=False, metaclass=abc.ABCMeta):
    """DEPRECATED Message subtype."""

    def __post_init__(
        self,
    ) -> None:
        warnings.warn(
            f"'{self.__class__.__name__}' was deprecated in DecLearn v2.4. "
            "It should no longer be used and may cause failures. It will be "
            "removed in DecLearn v2.6 and/or v3.0",
            DeprecationWarning,
        )


@dataclasses.dataclass
class Empty(DeprecatedMessage):
    """DEPRECATED empty message class."""

    typekey = "empty"


@dataclasses.dataclass
class GetMessageRequest(DeprecatedMessage):
    """DEPRECATED message-retrieval query message class."""

    typekey = "get_message"

    timeout: Optional[int] = None


@dataclasses.dataclass
class JoinRequest(DeprecatedMessage):
    """DEPRECATED process joining query message class."""

    typekey = "join_request"

    name: str
    data_info: Dict[str, Any]
    version: Optional[str] = None


@dataclasses.dataclass
class JoinReply(DeprecatedMessage):
    """DEPRECATED process joining reply message class."""

    typekey = "join_reply"

    accept: bool
    flag: str


def parse_message_from_string(
    string: str,
) -> Message:
    """DEPRECATED - Instantiate a Message from its serialized string.

    This function was DEPRECATED in DecLearn 2.4 and will be removed
    in v2.6 and/or v3.0. Use the `declearn.messaging.SerializedMessage`
    API to parse serialized message strings.

    Parameters
    ----------
    string:
        Serialized string dump of the message.

    Returns
    -------
    message:
        Message instance recovered from the input string.

    Raises
    ------
    KeyError
        If the string's typekey does not match any supported Message
        subclass.
    TypeError
        If the string cannot be parsed to identify a message typekey.
    ValueError
        If the serialized data fails to be properly decoded.
    """
    warnings.warn(
        "'parse_message_from_string' was deprecated in DecLearn 2.4, in "
        "favor of using 'declearn.messaging.SerializedMessage' to parse "
        "and deserialize 'Message' instances from strings. It will be "
        "removed in DecLearn version 2.6 and/or 3.0.",
        DeprecationWarning,
    )
    serialized = SerializedMessage.from_message_string(
        string
    )  # type: SerializedMessage[Any]
    return serialized.deserialize()
