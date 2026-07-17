# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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
from typing import Any, Dict, List, Optional, Tuple

from declearn.aggregator import Aggregator, ModelUpdates
from declearn.messaging._api import Message
from declearn.metrics import MetricInputType, MetricState
from declearn.model.api import Model, Vector
from declearn.optimizer import Optimizer
from declearn.optimizer.modules import AuxVar

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
    """Empty message used to ping or signal message reception.

    Fields
    ------
    reason:
        String that gives the reason behind the training cancel.
    """

    typekey = "cancel"

    reason: str


@dataclasses.dataclass
class Error(Message):
    """Error message container, used to convey exceptions between nodes.

    Fields
    ------
    message:
        Conveyed error message string.
    """

    typekey = "error"

    message: str


@dataclasses.dataclass
class EvaluationRequest(Message):
    """Server-emitted request to participate in an evaluation round.

    Fields
    ------
    round_i:
        Index of the evaluation round.
    weights:
        Model weights.
    batches:
        Dictionary mapping batches-generation parameters (in evaluation)
        to their value.
    n_steps:
        Maximum number of local evaluation steps to perform.
    timeout:
        Time (in seconds) beyond which to interrupt evaluation,
        regardless of the actual number of steps taken (> 0).
    """

    typekey = "eval_request"

    round_i: int
    weights: Optional[Vector]
    batches: Dict[str, Any]
    n_steps: Optional[int]
    timeout: Optional[int]


@dataclasses.dataclass
class EvaluationReply(Message):
    """Client-emitted results from a local evaluation round.

    Fields
    ------
    loss:
        Evaluation loss value.
    n_steps:
        Number of evaluation steps completed.
    t_spent:
        Time spent running evaluation steps (in seconds).
    metrics:
        Computed metrics, as partial values that may be shared with other
        agents to federatively compute final values.
    """

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
    """Generic message format, with action/params pair.

    Fields
    ------
    action:
        String that indicates the action to perform.
    params:
        Key-value parameters conveyed by the generic message.
    """

    typekey = "generic"

    action: str  # revise: Literal on possible flags?
    params: Dict[str, Any]


@dataclasses.dataclass
class InitRequest(Message):
    """Server-emitted request to initialize local model and optimizer.

    Fields
    ------
    model:
        Model initialized by the server.
    optim:
        Client optimizer initialized by the server, transfered to clients.
    aggrg:
        Aggregator initialized by the server.
    metrics:
        List of metric-like items (`MetricInputType`) involved in the federated
        process.
    dpsgd:
        True if privacy (through DP-SGD) is enabled in the federated process.
    secagg:
        True if secure aggregation is enabled in the federated process.
    fairness:
        True if fairness is enabled in the federated process.
    """

    typekey = "init_request"

    model: Model
    optim: Optimizer
    aggrg: Aggregator
    metrics: List[MetricInputType] = dataclasses.field(default_factory=list)
    dpsgd: bool = False
    secagg: Optional[str] = None
    fairness: bool = False

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        kwargs["model"] = self.model
        kwargs["optim"] = self.optim
        kwargs["aggrg"] = self.aggrg
        kwargs["metrics"] = self.metrics
        kwargs["dpsgd"] = self.dpsgd
        kwargs["secagg"] = self.secagg
        kwargs["fairness"] = self.fairness
        return kwargs


@dataclasses.dataclass
class InitReply(Message):
    """Client-emitted message indicating that initialization went fine."""

    typekey = "init_reply"


@dataclasses.dataclass
class MetadataQuery(Message):
    """Server-emitted request for metadata on a client's dataset.

    Fields
    ------
    fields:
        List of dataset metadata fields requested to the client.
    """

    typekey = "metadata_query"

    fields: List[str]


@dataclasses.dataclass
class MetadataReply(Message):
    """Client-emitted metadata in response to a server request.

    Fields
    ------
    data_info:
        Dictionary mapping metadata names (fields) to their values, sent back
        by a client to the server.
    """

    typekey = "metadata_reply"

    data_info: Dict[str, Any]


@dataclasses.dataclass
class PrivacyRequest(Message):
    """Server-emitted request to set up local differential privacy.

    Fields
    ------
    budget:
        Target total privacy budget per client, expressed in terms of
        (epsilon-delta)-DP over the full training schedule.
    sclip_norm:
        Clipping threshold of sample-wise gradients' euclidean norm.
        This parameter binds the sensitivity of sample-wise gradients.
    accountant:
        Accounting mechanism string used to estimate epsilon by Opacus.
    use_csprng:
        Whether to use cryptographically-secure pseudo-random numbers
        (CSPRNG) rather than the default numpy generator.
    seed:
        Optional seed to the noise-addition module's RNG.
    rounds:
        Maximum number of training and validation rounds to perform.
    batches:
        Dictionary mapping batches-generation parameters to their value.
    n_epoch:
        Maximum number of local data-processing epochs to perform.
    n_steps:
        Maximum number of local data-processing steps to perform.
    """

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
    """Server-emitted notification that the training process is over.

    Fields
    ------
    weights:
        Best global model weights.
    loss:
        Best global model loss.
    rounds:
        Number of training rounds that occurred in the process.
    """

    typekey = "stop_training"

    weights: Vector
    loss: float
    rounds: int


@dataclasses.dataclass
class TrainRequest(Message):
    """Server-emitted request to participate in a training round.

    Fields
    ------
    round_i:
        Index of the training round.
    weights:
        Model weights.
    aux_var:
        Dictionary mapping auxiliary variable names to the corresponding
        `AuxVar` instance.
    batches:
        Dictionary mapping batches-generation parameters (in training) to their
        value.
    n_epoch:
        Maximum number of local data-processing epochs to perform.
    n_steps:
        Maximum number of local data-processing steps to perform.
    timeout:
        Time (in seconds) beyond which to interrupt processing, regardless of
        the actual number of steps taken (> 0).
    """

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
    """Client-emitted results from a local training round.

    Fields
    ------
    n_epoch:
        Number of training epochs completed.
    n_steps:
        Number of training steps completed.
    t_spent:
        Time spent running training steps (in seconds).
    updates:
        Client model updates to transfer to the server.
    aux_var:
        Dictionary mapping auxiliary variable names to the corresponding
        `AuxVar` instance, to send back to the server.
    """

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
