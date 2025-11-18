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

"""Messages for the default Federated Learning process of DecLearn."""

import dataclasses
from typing import Any, Dict, List, Optional, Self, Tuple

from declearn.aggregator import Aggregator, ModelUpdates
from declearn.messaging._api import Message
from declearn.metrics import MetricInputType, MetricState
from declearn.model.api import Model, Vector
from declearn.optimizer import Optimizer
from declearn.optimizer.modules import AuxVar
from declearn.utils import deserialize_object, serialize_object

__all__ = [
    "CancelTraining",
    "Error",
    "EvaluationReply",
    "EvaluationRequest",
    "GenericMessage",
    "InitRequest",
    "InitReply",
    "MetadataQuery",
    "MetadataReply",
    "PrivacyRequest",
    "PrivacyReply",
    "StopTraining",
    "TrainReply",
    "TrainRequest",
]


@dataclasses.dataclass
class CancelTraining(Message):
    """Empty message used to ping or signal message reception."""

    typekey = "cancel"

    reason: str


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
    weights: Optional[Vector]
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
    metrics: Dict[str, MetricState] = dataclasses.field(default_factory=dict)

    def to_kwargs(
        self,
    ) -> Dict[str, Any]:
        # Undo recursive dict-conversion of dataclasses.
        kwargs = super().to_kwargs()
        kwargs["metrics"] = self.metrics
        return kwargs


@dataclasses.dataclass
class GenericMessage(Message):
    """Generic message format, with action/params pair."""

    typekey = "generic"

    action: str  # revise: Literal on possible flags?
    params: Dict[str, Any]


@dataclasses.dataclass
class InitRequest(Message):
    """Server-emitted request to initialize local model and optimizer."""

    typekey = "init_request"

    model: Model
    optim: Optimizer
    aggrg: Aggregator
    metrics: List[MetricInputType] = dataclasses.field(default_factory=list)
    dpsgd: bool = False
    secagg: Optional[str] = None
    fairness: bool = False

    def to_kwargs(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        data["model"] = serialize_object(self.model, group="Model").to_dict()
        data["optim"] = self.optim.get_config()
        data["aggrg"] = serialize_object(self.aggrg, "Aggregator").to_dict()
        data["metrics"] = self.metrics
        data["dpsgd"] = self.dpsgd
        data["secagg"] = self.secagg
        data["fairness"] = self.fairness
        return data

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> Self:
        kwargs["model"] = deserialize_object(kwargs["model"])
        kwargs["optim"] = Optimizer.from_config(kwargs["optim"])
        kwargs["aggrg"] = deserialize_object(kwargs["aggrg"])
        return cls(**kwargs)


@dataclasses.dataclass
class InitReply(Message):
    """Client-emitted message indicating that initialization went fine."""

    typekey = "init_reply"


@dataclasses.dataclass
class MetadataQuery(Message):
    """Server-emitted request for metadata on a client's dataset."""

    typekey = "metadata_query"

    fields: List[str]


@dataclasses.dataclass
class MetadataReply(Message):
    """Client-emitted metadata in response to a server request."""

    typekey = "metadata_reply"

    data_info: Dict[str, Any]


@dataclasses.dataclass
class PrivacyRequest(Message):
    """Server-emitted request to set up local differential privacy."""

    # dataclass; pylint: disable=too-many-instance-attributes

    typekey = "privacy_request"

    # PrivacyConfig
    budget: Tuple[float, float]
    sclip_norm: float
    accountant: str
    use_csprng: bool
    seed: Optional[int]
    # TrainingConfig + rounds
    rounds: int
    batches: Dict[str, Any]
    n_epoch: Optional[int]
    n_steps: Optional[int]


@dataclasses.dataclass
class PrivacyReply(Message):
    """Client-emitted message indicating that DP setup went fine."""

    typekey = "privacy_reply"


@dataclasses.dataclass
class StopTraining(Message):
    """Server-emitted notification that the training process is over."""

    typekey = "stop_training"

    weights: Vector
    loss: float
    rounds: int


@dataclasses.dataclass
class TrainRequest(Message):
    """Server-emitted request to participate in a training round."""

    typekey = "train_request"

    round_i: int
    weights: Optional[Vector]
    aux_var: Dict[str, AuxVar]
    batches: Dict[str, Any]
    n_epoch: Optional[int] = None
    n_steps: Optional[int] = None
    timeout: Optional[int] = None

    def to_kwargs(self) -> Dict[str, Any]:
        # Undo recursive dict-conversion of dataclasses.
        data = super().to_kwargs()
        data["aux_var"] = self.aux_var
        return data


@dataclasses.dataclass
class TrainReply(Message):
    """Client-emitted results from a local training round."""

    typekey = "train_reply"

    n_epoch: int
    n_steps: int
    t_spent: float
    updates: ModelUpdates
    aux_var: Dict[str, AuxVar]

    def to_kwargs(self) -> Dict[str, Any]:
        # Undo recursive dict-conversion of dataclasses.
        data = super().to_kwargs()
        data["updates"] = self.updates
        data["aux_var"] = self.aux_var
        return data
