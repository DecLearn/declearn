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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from declearn.aggregator import Aggregator
from declearn.communication.api import NetworkClient, NetworkServer
from declearn.communication.utils import (
    verify_client_messages_validity,
    verify_server_message_validity,
)
from declearn.fairness.api._messages import (
    FairnessCounts,
    FairnessGroups,
    FairnessRoundQuery,
    SecaggFairnessCounts,
)
from declearn.fairness.core import FairnessDataset
from declearn.messaging import Error, Message, SerializedMessage
from declearn.secagg.api import Decrypter, Encrypter
from declearn.secagg.messaging import aggregate_secagg_messages
from declearn.training import TrainingManager

__all__ = [
    "FairnessControllerClient",
    "FairnessControllerServer",
    "FairnessSetupQuery",
]


class FairnessControllerClient(metaclass=abc.ABCMeta):
    """Abstract base class for client-side fairness controllers."""

    def __init__(
        self,
    ) -> None:
        """Instantiate the client-side fairness controller."""
        self.groups = []  # type: List[Tuple[Any, ...]]

    async def setup_fairness(
        self,
        netwk: NetworkClient,
        manager: TrainingManager,
        secagg: Optional[Encrypter],
        params: Dict[str, Any],
    ) -> TrainingManager:
        """Participate in a routine to initialize fairness-aware learning.

        This routine has the following structure:

        - Exchange with the server to agree on an ordered list of sensitive
          groups defined by the interesection of 1+ sensitive attributes
          and (opt.) a classification target label.
        - Send (encrypted) group-wise training sample counts, that the server
          is to (secure-)aggregate.
        - Perform any additional actions specific to the algorithm in use.
            - On the client side, optionally alter the `TrainingManager` used.
            - On the server side, optionally alter the `Aggregator` used.

        Parameters
        ----------
        netwk:
            NetworkClient endpoint, registered to a server.
        manager:
            TrainingManager instance that was set up notwithstanding fairness.
        secagg:
            Optional SecAgg encryption controller.
        params:
            Dict of algorithm-specific keyword arguments received from
            the server as part of the query that triggered this routine.

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
        return await self.finalize_fairness_setup(
            netwk, manager, secagg, params
        )

    @abc.abstractmethod
    async def finalize_fairness_setup(
        self,
        netwk: NetworkClient,
        manager: TrainingManager,
        secagg: Optional[Encrypter],
        params: Dict[str, Any],
    ) -> TrainingManager:
        """Finalize the fairness setup routine and return an Aggregator.

        This method is called as part of `setup_fairness`, and should
        be defined by concrete subclasses to implement setup behavior
        once the initial query/reply messages have been exchanged.

        The returned `TrainingManager` may either be the input `manager`
        or a new or modified version of it, depending on the needs of
        the fairness-aware federated learning process being implemented.

        Parameters
        ----------
        netwk:
            NetworkClient endpoint, registered to a server.
        manager:
            TrainingManager instance that was set up notwithstanding fairness.
        secagg:
            Optional SecAgg encryption controller.
        params:
            Dict of algorithm-specific keyword arguments received from
            the server as part of the query that triggered this routine.

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
        """Participate in a round of actions to enforce fairness.

        Parameters
        ----------
        netwk:
            NetworkClient endpoint instance, connected to a server.
        manager:
            TrainingManager instance holding the local model, optimizer, etc.
            This method may (and usually does) have side effects on this.
        received:
            Serialized query message to participated in a fairness round.
        secagg:
            Optional SecAgg encryption controller.
        """


@dataclasses.dataclass
class FairnessSetupQuery(Message, register=False, metaclass=abc.ABCMeta):
    """ABC message for all Fairness setup init requests.

    This message should be subclassed into algorithm-specific messages.
    """

    @abc.abstractmethod
    def instantiate_controller(
        self,
    ) -> FairnessControllerClient:
        """Instantiate a `FairnessControllerClient` matching this query."""

    def get_setup_params(
        self,
    ) -> Dict[str, Any]:
        """Return a dict of parameters to pass to the client setup routine."""
        return {}


class FairnessControllerServer(metaclass=abc.ABCMeta):
    """Abstract base class for server-side fairness controllers."""

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
        """Orchestrate a routine to initialize fairness-aware learning.

        This routine has the following structure:

        - Send a setup query to clients, the type of which depends
          on the actual fairness-enforcing algorithm used.
        - Exchange with clients to agree on an ordered list of sensitive
          groups defined by the interesection of 1+ sensitive attributes
          and (opt.) a classification target label.
        - Receive and (secure-)aggregate group-wise sample counts across
          clients' training dataset.
        - Perform any additional actions specific to the algorithm in use.
            - On the server side, optionally alter the `Aggregator` used.
            - On the client side, optionally alter the `TrainingManager` used.

        Parameters
        ----------
        netwk:
            NetworkServer endpoint, to which clients are registered.
        aggregator:
            Aggregator instance that was set up notwithstanding fairness.
        secagg:
            Optional SecAgg decryption controller.

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
        round_i: int,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
    ) -> None:
        """Orchestrate a round of actions to enforce fairness.

        Parameters
        ----------
        round_i:
            Index of the current round (reflecting that of an upcoming
            training round).
        netwk:
            NetworkServer endpoint instance, to which clients are registered.
        secagg:
            Optional SecAgg decryption controller.
        """
