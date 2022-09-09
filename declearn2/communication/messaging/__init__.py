# coding: utf-8

"""Submodule defining messaging containers and flags used in declearn."""

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
    StopTraining,
    TrainReply,
    TrainRequest,
    parse_message_from_string,
)
