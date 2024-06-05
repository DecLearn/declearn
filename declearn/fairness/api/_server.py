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

"""Server-side ABC for fairness-aware federated learning controllers."""

import abc
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np

from declearn.aggregator import Aggregator
from declearn.communication.api import NetworkServer
from declearn.communication.utils import verify_client_messages_validity
from declearn.messaging import (
    FairnessCounts,
    FairnessGroups,
    FairnessSetupQuery,
    SerializedMessage,
)
from declearn.secagg.api import Decrypter
from declearn.secagg.messaging import (
    aggregate_secagg_messages,
    SecaggFairnessCounts,
)
from declearn.utils import create_types_registry, register_type

__all__ = [
    "FairnessControllerServer",
]


@create_types_registry(name="FairnessControllerServer")
class FairnessControllerServer(metaclass=abc.ABCMeta):
    """Abstract base class for server-side fairness controllers."""

    algorithm: ClassVar[str]
    """Name of the fairness-enforcing algorithm.

    This name should be unique across 'FairnessControllerServer' classes,
    and shared with a unique paired 'FairnessControllerClient'. It is used
    for type-registration and to enable instructing clients to instantiate
    a controller matching that chosen by the server in a federated setting.
    """

    def __init_subclass__(
        cls,
        register: bool = True,
    ) -> None:
        """Automatically type-register subclasses."""
        if register:
            register_type(cls, cls.algorithm, group="FairnessControllerServer")

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

        - Send a setup query to clients, resulting in the instantiation
          of client-side controllers matching this one.
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
        self.groups = await self._exchange_sensitive_groups_list(netwk)
        # Receive, (secure-)aggregate and return group-wise sample counts.
        counts = await self._aggregate_sensitive_groups_counts(netwk, secagg)
        # Run additional algorithm-specific setup steps.
        return await self.finalize_fairness_setup(netwk, counts, aggregator)

    def prepare_fairness_setup_query(
        self,
    ) -> FairnessSetupQuery:
        """Return a request to setup fairness, broadcastable to clients.

        Returns
        -------
        message:
            `FairnessSetupQuery` instance to be sent to clients in order
            to trigger the Fairness setup protocol.
        """
        return FairnessSetupQuery(
            algorithm=self.algorithm,
            params={"f_type": self.f_type, "f_args": self.f_args},
        )

    @staticmethod
    async def _exchange_sensitive_groups_list(
        netwk: NetworkServer,
    ) -> List[Tuple[Any, ...]]:
        """Receive, aggregate, share and return sensitive group definitions."""
        received = await netwk.wait_for_messages()
        # Verify and deserialize client-wise sensitive group definitions.
        messages = await verify_client_messages_validity(
            netwk, received, expected=FairnessGroups
        )
        # Gather the sorted union of all existing definitions.
        unique = {group for msg in messages.values() for group in msg.groups}
        groups = sorted(list(unique))
        # Send it to clients, and expect their reply (encrypted counts).
        await netwk.broadcast_message(FairnessGroups(groups=groups))
        return groups

    async def _aggregate_sensitive_groups_counts(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
    ) -> List[int]:
        """Receive, (secure-)aggregate and return group-wise sample counts."""
        received = await netwk.wait_for_messages()
        if secagg is None:
            return await self._aggregate_sensitive_groups_counts_cleartext(
                netwk=netwk, received=received, n_groups=len(self.groups)
            )
        return await self._aggregate_sensitive_groups_counts_encrypted(
            netwk=netwk, received=received, decrypter=secagg
        )

    @staticmethod
    async def _aggregate_sensitive_groups_counts_cleartext(
        netwk: NetworkServer,
        received: Dict[str, SerializedMessage],
        n_groups: int,
    ) -> List[int]:
        """Deserialize and aggregate cleartext group-wise counts."""
        replies = await verify_client_messages_validity(
            netwk, received, expected=FairnessCounts
        )
        counts = np.zeros(n_groups, dtype="uint64")
        for message in replies.values():
            counts = counts + np.asarray(message.counts, dtype="uint64")
        return counts.tolist()

    @staticmethod
    async def _aggregate_sensitive_groups_counts_encrypted(
        netwk: NetworkServer,
        received: Dict[str, SerializedMessage],
        decrypter: Decrypter,
    ) -> List[int]:
        """Deserialize and secure-aggregate encrypted group-wise counts."""
        replies = await verify_client_messages_validity(
            netwk, received, expected=SecaggFairnessCounts
        )
        aggregated = aggregate_secagg_messages(replies, decrypter)
        return aggregated.counts

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
    async def finalize_fairness_round(
        self,
        round_i: int,
        values: List[float],
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Orchestrate a round of actions to enforce fairness.

        This method is designed to be called after an initial query
        has been sent and responded to by clients, resulting in the
        federated computation of fairness(-related) metrics.

        Parameters
        ----------
        round_i:
            Index of the current round (reflecting that of an upcoming
            training round).
        values:
            Aggregated metrics resulting from the fairness evaluation
            run by clients at this round.
        netwk:
            NetworkServer endpoint instance, to which clients are registered.
        secagg:
            Optional SecAgg decryption controller.

        Returns
        -------
        metrics:
            Computed local fairness(-related) metrics computed as part
            of this routine, as a dict mapping scalar or numpy array
            values with their name.
        """
