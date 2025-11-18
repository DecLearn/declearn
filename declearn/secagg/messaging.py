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

"""SecAgg counterparts to some default Federated Learning messages."""

import abc
import dataclasses
from typing import Dict, Generic, List, Mapping, Self, TypeVar

from declearn.aggregator import ModelUpdates
from declearn.messaging import (
    EvaluationReply,
    FairnessCounts,
    FairnessReply,
    Message,
    TrainReply,
)
from declearn.metrics import MetricState
from declearn.optimizer.modules import AuxVar
from declearn.secagg.api import Decrypter, Encrypter, SecureAggregate

__all__ = [
    "SecaggEvaluationReply",
    "SecaggFairnessCounts",
    "SecaggFairnessReply",
    "SecaggMessage",
    "SecaggTrainReply",
    "aggregate_secagg_messages",
]


MessageT = TypeVar("MessageT", bound=Message)


@dataclasses.dataclass
class SecaggMessage(
    Generic[MessageT],
    Message,
    register=False,
    metaclass=abc.ABCMeta,
):
    """Abstract 'Message' subclass for SecAgg-protected client results.

    This 'Message' subclass implements a specific API for SecAgg-encrypted
    data, with dedicated methods to transform (from and to) cleartext base
    messages, and the possibility to (secure-)aggregate clients' protected
    results by using the `SecaggMessage.aggregate` method, which involves
    providing with either an `Encrypter` or `Decrypter` decrypter.
    """

    @classmethod
    @abc.abstractmethod
    def from_cleartext_message(
        cls,
        cleartext: MessageT,
        encrypter: Encrypter,
    ) -> Self:
        """Encrypt a cleartext Message into its SecaggMessage counterpart.

        Parameters
        ----------
        cleartext:1
            Message that needs encryption prior to sharing.
        encrypter:
            Controller to be used for message contents' encryption.
        """

    @abc.abstractmethod
    def decrypt_wrapped_message(
        self,
        decrypter: Decrypter,
    ) -> MessageT:
        """Decrypt a SecaggMessage into its cleartext Message counterpart.

        This method requires (and does not verify) that the proper number
        of client-wise encrypted messages have been sum-aggregated into
        the one that is being decrypted.

        Parameters
        ----------
        decrypter:
            Controller to be used for aggregated contents' decryption.
        """

    @abc.abstractmethod
    def aggregate(
        self,
        other: Self,
        decrypter: Decrypter,
    ) -> Self:
        """Aggregate two clients' SecaggMessage instances into one."""


def aggregate_secagg_messages(
    messages: Mapping[str, SecaggMessage[MessageT]],
    decrypter: Decrypter,
) -> MessageT:
    """Secure-Aggregate (and decrypt) client-issued encrypted messages.

    Parameters
    ----------
    messages:
        Mapping of client-wise `SecaggMessage` instances, wrapping
        similar messages that need secure aggregation.
    decrypter:
        Decryption controller to use when aggregating inputs.

    Returns
    -------
    message:
        Cleartext message resulting from the secure aggregation
        of input `messages`.
    """
    encrypted = list(messages.values())
    aggregate = encrypted[0]
    for message in encrypted[1:]:
        aggregate = aggregate.aggregate(message, decrypter=decrypter)
    return aggregate.decrypt_wrapped_message(decrypter=decrypter)


@dataclasses.dataclass
class SecaggTrainReply(SecaggMessage[TrainReply]):
    """SecAgg-wrapped 'TrainReply' message."""

    typekey = "secagg_train_reply"

    n_steps: int
    t_spent: float
    updates: SecureAggregate[ModelUpdates]
    aux_var: Dict[str, SecureAggregate[AuxVar]]

    @classmethod
    def from_cleartext_message(
        cls,
        cleartext: TrainReply,
        encrypter: Encrypter,
    ) -> Self:
        n_steps = encrypter.encrypt_uint(cleartext.n_steps)
        t_spent = cleartext.t_spent
        updates = encrypter.encrypt_aggregate(cleartext.updates)
        aux_var = {
            key: encrypter.encrypt_aggregate(val)
            for key, val in cleartext.aux_var.items()
        }
        return cls(
            n_steps=n_steps, t_spent=t_spent, updates=updates, aux_var=aux_var
        )

    def decrypt_wrapped_message(
        self,
        decrypter: Decrypter,
    ) -> TrainReply:
        n_steps = decrypter.decrypt_uint(self.n_steps)
        t_spent = self.t_spent
        updates = decrypter.decrypt_aggregate(self.updates)
        aux_var = {
            key: decrypter.decrypt_aggregate(val)
            for key, val in self.aux_var.items()
        }
        return TrainReply(
            n_epoch=-1,
            n_steps=n_steps,
            t_spent=t_spent,
            updates=updates,
            aux_var=aux_var,
        )

    def aggregate(
        self,
        other: Self,
        decrypter: Decrypter,
    ) -> Self:
        n_steps = decrypter.sum_encrypted([self.n_steps, other.n_steps])
        t_spent = max(self.t_spent, other.t_spent)
        updates = self.updates + other.updates
        if set(self.aux_var).symmetric_difference(other.aux_var):
            raise KeyError(
                "Cannot aggregate SecAgg-protected auxiliary variables "
                "with distinct keys."
            )
        aux_var = {
            key: val + other.aux_var[key] for key, val in self.aux_var.items()
        }
        return self.__class__(
            n_steps=n_steps, t_spent=t_spent, updates=updates, aux_var=aux_var
        )


@dataclasses.dataclass
class SecaggEvaluationReply(SecaggMessage[EvaluationReply]):
    """SecAgg-wrapped 'EvaluationReply' message."""

    typekey = "secagg_eval_reply"

    loss: int
    n_steps: int
    t_spent: float
    metrics: Dict[str, SecureAggregate[MetricState]]

    @classmethod
    def from_cleartext_message(
        cls,
        cleartext: EvaluationReply,
        encrypter: Encrypter,
    ) -> Self:
        loss = encrypter.encrypt_float(cleartext.loss)
        n_steps = encrypter.encrypt_uint(cleartext.n_steps)
        t_spent = cleartext.t_spent
        metrics = {
            key: encrypter.encrypt_aggregate(val)
            for key, val in cleartext.metrics.items()
        }
        return cls(
            loss=loss, n_steps=n_steps, t_spent=t_spent, metrics=metrics
        )

    def decrypt_wrapped_message(
        self,
        decrypter: Decrypter,
    ) -> EvaluationReply:
        loss = decrypter.decrypt_float(self.loss)
        n_steps = decrypter.decrypt_uint(self.n_steps)
        t_spent = self.t_spent
        metrics = {
            key: decrypter.decrypt_aggregate(val)
            for key, val in self.metrics.items()
        }
        return EvaluationReply(
            loss=loss, n_steps=n_steps, t_spent=t_spent, metrics=metrics
        )

    def aggregate(
        self,
        other: Self,
        decrypter: Decrypter,
    ) -> Self:
        loss = decrypter.sum_encrypted([self.loss, other.loss])
        n_steps = decrypter.sum_encrypted([self.n_steps, other.n_steps])
        t_spent = max(self.t_spent, other.t_spent)
        if set(self.metrics).symmetric_difference(other.metrics):
            raise KeyError(
                "Cannot aggregate SecAgg-protected metric states with "
                "distinct keys."
            )
        metrics = {
            key: val + other.metrics[key] for key, val in self.metrics.items()
        }
        return self.__class__(
            loss=loss, n_steps=n_steps, t_spent=t_spent, metrics=metrics
        )


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
            for v_a, v_b in zip(self.counts, other.counts, strict=False)
        ]
        return self.__class__(counts=counts)


@dataclasses.dataclass
class SecaggFairnessReply(SecaggMessage[FairnessReply]):
    """SecAgg-wrapped 'FairnessReply' message."""

    typekey = "secagg_fairness_reply"

    values: List[int]

    @classmethod
    def from_cleartext_message(
        cls,
        cleartext: FairnessReply,
        encrypter: Encrypter,
    ) -> Self:
        values = [encrypter.encrypt_float(value) for value in cleartext.values]
        return cls(values=values)

    def decrypt_wrapped_message(
        self,
        decrypter: Decrypter,
    ) -> FairnessReply:
        values = [decrypter.decrypt_float(value) for value in self.values]
        return FairnessReply(values=values)

    def aggregate(
        self,
        other: Self,
        decrypter: Decrypter,
    ) -> Self:
        if len(self.values) != len(other.values):
            raise ValueError(
                "Cannot aggregate SecAgg-protected fairness values with "
                "distinct shapes."
            )
        values = [
            decrypter.sum_encrypted([v_a, v_b])
            for v_a, v_b in zip(self.values, other.values, strict=False)
        ]
        return self.__class__(values=values)
