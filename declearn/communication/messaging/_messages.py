# coding: utf-8

"""Dataclasses defining messages used in declearn communications."""

import dataclasses
import json
from abc import ABCMeta
from typing import Any, Dict, List, Optional, Type


from declearn.model.api import Model, NumpyVector, Vector
from declearn.optimizer import Optimizer
from declearn.utils import (
    deserialize_object, json_pack, json_unpack, serialize_object
)


__all__ = [
    'CancelTraining',
    'Empty',
    'Error',
    'EvaluationReply',
    'EvaluationRequest',
    'GenericMessage',
    'GetMessageRequest',
    'InitRequest',
    'JoinReply',
    'JoinRequest',
    'Message',
    'StopTraining',
    'TrainReply',
    'TrainRequest',
    'parse_message_from_string',
]


@dataclasses.dataclass
class Message(metaclass=ABCMeta):
    """Base class to define declearn messages."""

    typekey: str = dataclasses.field(init=False)

    def to_string(self) -> str:
        """Convert the message to a JSON-serialized string."""
        data = dataclasses.asdict(self)
        return json.dumps(data, default=json_pack)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> 'Message':
        """Parse the message from JSON-deserialized attributes."""
        # NOTE: override this method to de-serialize attributes
        #       that are not handled by declearn.utils.json_pack
        return cls(**kwargs)


@dataclasses.dataclass
class CancelTraining(Message):
    """Empty message used to ping or signal message reception."""

    typekey = "cancel"

    reason: str


@dataclasses.dataclass
class Empty(Message):
    """Empty message used to ping or signal message reception."""

    typekey = "empty"


@dataclasses.dataclass
class Error(Message):
    """Error message container, used to convey exceptions between nodes."""

    typekey = "error"

    message: str


@dataclasses.dataclass
class EvaluationRequest(Message):
    """Server-emitted request to participate in an evaluation round."""

    typekey = "eval_request"

    round_i: int
    weights: NumpyVector
    batches: Dict[str, Any]
    n_steps: Optional[int]
    timeout: Optional[int]


@dataclasses.dataclass
class EvaluationReply(Message):
    """Client-emitted results from a local evaluation round."""

    typekey = "eval_reply"

    loss: float
    n_steps: int
    t_spent: float


@dataclasses.dataclass
class GenericMessage(Message):
    """Generic message format, with action/params pair."""

    typekey = "generic"

    action: str  # revise: Literal on possible flags?
    params: Dict[str, Any]


@dataclasses.dataclass
class GetMessageRequest(Message):
    """Client-emitted prompt to collect a message posted by the server."""

    typekey = "get_message"

    timeout: Optional[int] = None


@dataclasses.dataclass
class InitRequest(Message):
    """Server-emitted request to initialize local model and optimizer."""

    typekey = "init_request"

    model: Model
    optim: Optimizer

    def to_string(self) -> str:
        data = dataclasses.asdict(self)
        data["model"] = serialize_object(self.model, group="Model").to_dict()
        data["optim"] = self.optim.get_config()
        return json.dumps(data, default=json_pack)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> 'Message':
        kwargs["model"] = deserialize_object(kwargs["model"])
        kwargs["optim"] = Optimizer.from_config(kwargs["optim"])
        return cls(**kwargs)


@dataclasses.dataclass
class JoinRequest(Message):
    """Client-emitted request to join training."""

    typekey = "join_request"

    name: str
    data_info: Dict[str, Any]


@dataclasses.dataclass
class JoinReply(Message):
    """Server-emitted reply to a JoinRequest."""

    typekey = "join_reply"

    accept: bool
    flag: str


@dataclasses.dataclass
class StopTraining(Message):
    """Server-emitted notification that the training process is over."""

    typekey = "stop_training"

    weights: NumpyVector
    loss: float
    rounds: int


@dataclasses.dataclass
class TrainRequest(Message):
    """Server-emitted request to participate in a training round."""

    typekey = "train_request"

    round_i: int
    weights: NumpyVector
    aux_var: Dict[str, Dict[str, Any]]
    batches: Dict[str, Any]
    n_epoch: Optional[int] = None
    n_steps: Optional[int] = None
    timeout: Optional[int] = None


@dataclasses.dataclass
class TrainReply(Message):
    """Client-emitted results from a local training round."""

    typekey = "train_reply"

    n_epoch: int
    n_steps: int
    t_spent: float
    updates: Vector
    aux_var: Dict[str, Dict[str, Any]]


_MESSAGE_CLASSES = [
    CancelTraining,
    Empty,
    Error,
    EvaluationReply,
    EvaluationRequest,
    GenericMessage,
    GetMessageRequest,
    InitRequest,
    JoinReply,
    JoinRequest,
    StopTraining,
    TrainReply,
    TrainRequest,
]  # type: List[Type[Message]]
MESSAGE_CLASSES = {
    cls.typekey: cls for cls in _MESSAGE_CLASSES
}


def parse_message_from_string(
        string: str,
    ) -> Message:
    """Instantiate a Message from a JSON-serialized string."""
    data = json.loads(string, object_hook=json_unpack)
    if "typekey" not in data:
        raise KeyError("Missing required 'typekey'")
    typekey = data.pop("typekey")
    cls = MESSAGE_CLASSES.get(typekey)
    if cls is None:
        raise KeyError(f"No Message matches typekey '{typekey}'.")
    return cls.from_kwargs(**data)
