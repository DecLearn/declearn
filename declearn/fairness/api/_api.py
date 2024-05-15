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

"""Draft API for Fairness-aware Federated Learning."""

import abc
import dataclasses
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.aggregator import Aggregator
from declearn.communication.api import NetworkClient, NetworkServer
from declearn.communication.utils import (
    verify_client_messages_validity,
    verify_server_message_validity,
)
from declearn.fairness.core import FairnessDataset
from declearn.main.utils import TrainingManager
from declearn.messaging import Error, Message, SerializedMessage
from declearn.secagg.api import Decrypter, Encrypter
from declearn.secagg.messaging import SecaggMessage, aggregate_secagg_messages

__all__ = [
    "FairnessAccuracy",
    "FairnessCounts",
    "FairnessControllerClient",
    "FairnessControllerServer",
    "FairnessGroups",
    "FairnessRoundQuery",
    "FairnessRoundReply",
    "FairnessSetupQuery",
    "SecaggFairnessAccuracy",
    "SecaggFairnessCounts",
]


@abc.abstractmethod
@dataclasses.dataclass
class FairnessSetupQuery(Message, register=False):
    """Abstract base Message for server-emitted fairness setup queries."""


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
            for v_a, v_b in zip(self.counts, other.counts)
        ]
        return self.__class__(counts=counts)


@dataclasses.dataclass
class FairnessAccuracy(Message):
    """Message for client-emitted model accuracy across sensitive groups.

    Fields
    ------
    values:
        List of group-wise accuracy values, ordered based
        on an agreed-upon sorted list of sensitive groups.
    """

    values: List[float]

    typekey = "fairness-accuracy"


@dataclasses.dataclass
class SecaggFairnessAccuracy(SecaggMessage[FairnessAccuracy]):
    """SecAgg counterpart of the 'FairnessAccuracy' message class."""

    values: List[int]

    typekey = "secagg-fairness-accuracy"

    @classmethod
    def from_cleartext_message(
        cls,
        cleartext: FairnessAccuracy,
        encrypter: Encrypter,
    ) -> Self:
        values = [encrypter.encrypt_float(val) for val in cleartext.values]
        return cls(values=values)

    def decrypt_wrapped_message(
        self,
        decrypter: Decrypter,
    ) -> FairnessAccuracy:
        values = [decrypter.decrypt_float(val) for val in self.values]
        return FairnessAccuracy(values=values)

    def aggregate(
        self,
        other: Self,
        decrypter: Decrypter,
    ) -> Self:
        values = [
            decrypter.sum_encrypted([v_a, v_b])
            for v_a, v_b in zip(self.values, other.values)
        ]
        return self.__class__(values=values)


@dataclasses.dataclass
class FairnessRoundQuery(Message):
    """Base Message for server-emitted fairness-computation queries.

    The base `FairnessRoundQuery` defines information that is used
    when evaluating a model's accuracy and/or loss over group-wise
    training samples.

    Subclasses may be defined to add algorithm-specific information.

    Fields
    ------
    batch_size:
        Number of samples per batch when computing metrics.
    n_batch:
        Optional maximum number of batches to draw per group.
        If None, use the entire wrapped dataset.
    thresh:
        Optional binarization threshold for binary classification
        models' output scores. If None, use 0.5 by default, or 0.0
        for `SklearnSGDModel` instances.
        Unused for multinomial classifiers (argmax over scores).
    """

    batch_size: int = 32
    n_batch: Optional[int] = None
    thresh: Optional[float] = None

    typekey = "fairness-round-query"


@dataclasses.dataclass
class FairnessRoundReply(Message):
    """Base Message for client-emitted fairness-round end signal.

    By default this message is empty, merely noticing that things
    went well. Subclasses may be used to convey algorithm-specific
    results or information.
    """

    typekey = "fairness-round-reply"


class FairnessControllerServer(metaclass=abc.ABCMeta):
    """Docstring."""

    def __init__(
        self,
        f_type: str,
        f_args: Optional[Dict[str, Any]],
    ) -> None:
        """Instantiate the server-side fairness controller.

        Parameters
        ----------
        f_type:
            Name of the fairness function to evaluate and optimize.
        f_args:
            Optional dict of keyword arguments to the fairness function.
        """
        self.f_type = f_type
        self.f_args = f_args or {}
        self.groups = []  # type: List[Tuple[Any, ...]]

    async def setup_fairness(
        self,
        netwk: NetworkServer,
        aggregator: Aggregator,
        secagg: Optional[Decrypter],
    ) -> Aggregator:
        """Docstring."""
        # Send a setup query to all clients.
        query = self.prepare_fairness_setup_query()
        await netwk.broadcast_message(query)
        # Receive, aggregate, assign and send back sensitive group definitions.
        await self._exchange_sensitive_groups_list(netwk)
        # Wait for group-wise sample counts from clients.
        received = await netwk.wait_for_messages()
        # When SecAgg is not used, expect cleartext group-wise counts.
        if secagg is None:
            replies = await verify_client_messages_validity(
                netwk, received, expected=FairnessCounts
            )
            counts = self._aggregate_cleartext_counts(replies)
        # When SecAgg is used, expect and secure-aggregate encrypted counts.
        else:
            sec_rep = await verify_client_messages_validity(
                netwk, received, expected=SecaggFairnessCounts
            )
            counts = aggregate_secagg_messages(sec_rep, secagg).counts
        # Run additional algorithm-specific setup steps.
        return await self.finalize_fairness_setup(netwk, counts, aggregator)

    def _aggregate_cleartext_counts(
        self,
        messages: Dict[str, FairnessCounts],
    ) -> List[int]:
        """Sum group-wise sample counts received from clients."""
        counts = np.zeros(len(self.groups), dtype="uint64")
        for message in messages.values():
            counts += np.asarray(message.counts, dtype="uint64")
        return counts.tolist()

    async def _exchange_sensitive_groups_list(
        self,
        netwk: NetworkServer,
    ) -> None:
        """Receive, aggregate, assign and share sensitive group definitions."""
        received = await netwk.wait_for_messages()
        # Verify and deserialize client-wise sensitive group definitions.
        messages = await verify_client_messages_validity(
            netwk, received, expected=FairnessGroups
        )
        # Gather the sorted union of all existing definitions.
        unique = {group for msg in messages.values() for group in msg.groups}
        self.groups = sorted(list(unique))
        # Send it to clients, and expect their reply (encrypted counts).
        await netwk.broadcast_message(FairnessGroups(groups=self.groups))

    @abc.abstractmethod
    def prepare_fairness_setup_query(
        self,
    ) -> FairnessSetupQuery:
        """Return a request to setup fairness, broadcastable to clients.

        Returns
        -------
        message:
            `FairnessSetupQuery` subclass instance to be sent to clients
            in order to trigger the Fairness setup protocol.
        """

    @abc.abstractmethod
    async def finalize_fairness_setup(
        self,
        netwk: NetworkServer,
        counts: List[int],
        aggregator: Aggregator,
    ) -> Aggregator:
        """Finalize the fairness setup routine and return an Aggregator.

        This method is called as part of `setup_fairness`, and should
        be defined by concrete subclasses to implement setup behavior
        once the initial query/reply messages have been exchanged.

        The returned `Aggregator` may either be the input `aggregator`
        or a new or modified version of it, depending on the needs of
        the fairness-aware federated learning process being implemented.

        Warns
        -----
        RuntimeWarning
            If the returned aggregator differs from the input one.

        Returns
        -------
        aggregator:
            `Aggregator` instance to use in the FL process, that may
            or may not have been altered compared with the input one.
        """

    @abc.abstractmethod
    async def fairness_round(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
    ) -> None:
        """Docstring."""


class FairnessControllerClient(metaclass=abc.ABCMeta):
    """Docstring."""

    setup_query_cls: ClassVar[Type[FairnessSetupQuery]]

    def __init__(
        self,
    ) -> None:
        """Docstring."""
        self.groups = []  # type: List[Tuple[Any, ...]]

    async def setup_fairness(
        self,
        netwk: NetworkClient,
        received: SerializedMessage,
        manager: TrainingManager,
        secagg: Optional[Encrypter],
    ) -> TrainingManager:
        """Docstring."""
        # Verify and unpack the received server query.
        query = await verify_server_message_validity(
            netwk, received, expected=self.setup_query_cls
        )
        # Verify that a training 'FairnessDataset' is available.
        if not isinstance(manager.train_data, FairnessDataset):
            msg = "Cannot set up fairness without a 'FairnessDataset'."
            await netwk.send_message(Error(msg))
            raise TypeError(msg)
        # Gather local sensitive groups and their sample counts.
        counts = manager.train_data.get_sensitive_group_counts()
        groups = list(counts)
        # Share them and receive a unified, ordered list of groups.
        await netwk.send_message(FairnessGroups(groups=groups))
        received = await netwk.recv_message()
        message = await verify_server_message_validity(
            netwk, received, expected=FairnessGroups
        )
        self.groups = message.groups
        # Sort and fill out sample counts, opt. encrypt them and send them.
        reply = FairnessCounts([counts.get(group, 0) for group in self.groups])
        if secagg is None:
            await netwk.send_message(reply)
        else:
            await netwk.send_message(
                SecaggFairnessCounts.from_cleartext_message(reply, secagg)
            )
        # Run additional algorithm-specific setup steps.
        return await self.finalize_fairness_setup(netwk, query, manager)

    @abc.abstractmethod
    async def finalize_fairness_setup(
        self,
        netwk: NetworkClient,
        query: FairnessSetupQuery,
        manager: TrainingManager,
    ) -> TrainingManager:
        """Finalize the fairness setup routine and return an Aggregator.

        This method is called as part of `setup_fairness`, and should
        be defined by concrete subclasses to implement setup behavior
        once the initial query/reply messages have been exchanged.

        The returned `TrainingManager` may either be the input `manager`
        or a new or modified version of it, depending on the needs of
        the fairness-aware federated learning process being implemented.

        Warns
        -----
        RuntimeWarning
            If the returned training manager differs from the input one.

        Returns
        -------
        manager:
            `TrainingManager` instance to use in the FL process, that may
            or may not have been altered compared with the input one.
        """

    @abc.abstractmethod
    async def fairness_round(
        self,
        netwk: NetworkClient,
        manager: TrainingManager,
        received: SerializedMessage[FairnessRoundQuery],
        secagg: Optional[Encrypter],
    ) -> None:
        """Docstring."""
