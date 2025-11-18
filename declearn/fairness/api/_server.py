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

"""Server-side ABC for fairness-aware federated learning controllers."""

import abc
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np

from declearn.aggregator import Aggregator
from declearn.communication.api import NetworkServer
from declearn.communication.utils import verify_client_messages_validity
from declearn.messaging import (
    Error,
    FairnessCounts,
    FairnessGroups,
    FairnessReply,
    FairnessSetupQuery,
    SerializedMessage,
)
from declearn.secagg.api import Decrypter
from declearn.secagg.messaging import (
    SecaggFairnessCounts,
    SecaggFairnessReply,
    aggregate_secagg_messages,
)
from declearn.utils import (
    access_registered,
    create_types_registry,
    register_type,
)

__all__ = [
    "FairnessControllerServer",
]


@create_types_registry(name="FairnessControllerServer")
class FairnessControllerServer(metaclass=abc.ABCMeta):
    """Abstract base class for server-side fairness controllers.

    Usage
    -----
    A `FairnessControllerServer` (subclass) instance has two main
    routines that are to be called as part of a federated learning
    process:

    - `setup_fairness`:
        This routine is to be called only once, during the setup of the
        overall federated learning task. It triggers the following process:
            - Send a `FairnessSetupQuery` to clients so that they
              instantiate a counterpart `FairnessControllerClient`.
            - Run a basic routine to exchange sensitive group definitions
              and (secure-)aggregate associated sample counts.
            - Perform any additional algorithm-specific setup actions.

    - `run_fairness_round`:
        This routine is to be called once per round, before the next
        training round occurs. A `FairnessQuery` should be sent to
        clients prior to calling it. It triggers the following process:
            - Run a basic routine to receive and (secure-)aggregate
              metrics computed by clients that relate to fairness.
            - Perform any additonal algorithm-specific round actions.

    Inheritance
    -----------
    Algorithm-specific subclasses should define the following abstract
    attribute and methods:

    - `algorithm`:
        Abstract string class attribute. Name under which this controller
        and its client-side counterpart classes are registered.
    - `finalize_fairness_setup`:
        Method implementing any algorithm-specific setup actions.
    - `finalize_fairness_round`:
        Method implementing any algorithm-specific round actions.

    By default, subclasses are type-registered under their `algorithm`
    name and "FairnessControllerServer" group upon declaration. This can
    be prevented by passing `register=False` to the inheritance parameters
    (e.g. `class Cls(FairnessControllerServer, register=False)`).
    See `declearn.utils.register_type` for details on types registration.
    """

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
        f_args: Optional[Dict[str, Any]] = None,
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
        self.groups: List[Tuple[Any, ...]] = []

    # Fairness Setup methods.

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
        # Agree on a list of sensitive groups and aggregate sample counts.
        counts = await self.exchange_sensitive_groups_list_and_counts(
            netwk, secagg
        )
        # Run additional algorithm-specific setup steps.
        return await self.finalize_fairness_setup(
            netwk, secagg, counts, aggregator
        )

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

    async def exchange_sensitive_groups_list_and_counts(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
    ) -> List[int]:
        """Agree on a list of sensitive groups and aggregate sample counts.

        This method performs the following routine:

        - Await `FairnessGroups` messages from clients with group definitions.
        - Assign a sorted list of sensitive groups as `groups` attribute.
        - Share that list with clients.
        - Await possibly-encrypted group-wise sample counts from clients.
        - (Secure-)Aggregate these sample counts and return them.

        Parameters
        ----------
        netwk:
            `NetworkServer` endpoint, through which a fairness setup query
            was previously sent to all clients.
        secagg:
            Optional SecAgg decryption controller.

        Returns
        -------
        counts:
            List of group-wise total sample count across clients,
            sorted based on the newly-assigned `self.groups`.
        """
        # Receive, aggregate, assign and send back sensitive group definitions.
        self.groups = await self._exchange_sensitive_groups_list(netwk)
        # Receive, (secure-)aggregate and return group-wise sample counts.
        return await self._aggregate_sensitive_groups_counts(netwk, secagg)

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
        counts: np.ndarray = np.zeros(n_groups, dtype="uint64")
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
        secagg: Optional[Decrypter],
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

    # Fairness Round methods.

    async def run_fairness_round(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Secure-aggregate and post-process fairness measures.

        This method is to be run **after** having sent a `FairnessQuery`
        to clients. It consists in receiving, (secure-)aggregating and
        post-processing measures that clients produce as a reply to that
        query. This may involve further algorithm-specific communications.

        Parameters
        ----------
        netwk:
            NetworkServer endpoint instance, to which clients are registered.
        secagg:
            Optional SecAgg decryption controller.

        Returns
        -------
        metrics:
            Fairness(-related) metrics computed as part of this routine,
            as a dict mapping scalar or numpy array values with their name.
        """
        values = await self.receive_and_aggregate_fairness_measures(
            netwk, secagg
        )
        return await self.finalize_fairness_round(netwk, secagg, values)

    async def receive_and_aggregate_fairness_measures(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
    ) -> List[float]:
        """Await and (secure-)aggregate client-wise fairness-related metrics.

        This method is designed to be called after sending a `FairnessQuery`
        to clients, and returns values that are yet to be parsed and used by
        the algorithm-dependent `finalize_fairness_round` method.

        Parameters
        ----------
        netwk:
            NetworkServer endpoint instance, to which clients are registered.
        secagg:
            Optional SecAgg decryption controller.

        Returns
        -------
        metrics:
            List of sum-aggregated fairness-related metrics (as floats).
            By default, these are group-wise accuracy values; this may
            however be changed or expanded by algorithm-specific classes.
        """
        received = await netwk.wait_for_messages()
        # Case when expecting cleartext values.
        if secagg is None:
            replies = await verify_client_messages_validity(
                netwk, received, expected=FairnessReply
            )
            if len(set(len(r.values) for r in replies.values())) != 1:
                error = "Clients sent fairness values of different lengths."
                await netwk.broadcast_message(Error(error))
                raise RuntimeError(error)
            return [
                sum(rval)
                for rval in zip(
                    *[reply.values for reply in replies.values()], strict=False
                )
            ]
        # Case when expecting encrypted values.
        secagg_replies = await verify_client_messages_validity(
            netwk, received, expected=SecaggFairnessReply
        )
        agg_reply = aggregate_secagg_messages(secagg_replies, decrypter=secagg)
        return agg_reply.values

    @abc.abstractmethod
    async def finalize_fairness_round(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
        values: List[float],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Orchestrate a round of actions to enforce fairness.

        This method is designed to be called after an initial query
        has been sent and responded to by clients, resulting in the
        federated computation of fairness(-related) metrics.

        Parameters
        ----------
        netwk:
            NetworkServer endpoint instance, to which clients are registered.
        secagg:
            Optional SecAgg decryption controller.
        values:
            Aggregated metrics resulting from the fairness evaluation
            run by clients at this round.

        Returns
        -------
        metrics:
            Fairness(-related) metrics computed as part of this routine,
            as a dict mapping scalar or numpy array values with their name.
        """

    @staticmethod
    def from_specs(
        algorithm: str,
        f_type: str,
        f_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "FairnessControllerServer":
        """Instantiate a 'FairnessControllerServer' from its specifications.

        Parameters
        ----------
        algorithm:
            Name of the algorithm associated with the target controller class.
        f_type:
            Name of the fairness function to evaluate and optimize.
        f_args:
            Optional dict of keyword arguments to the fairness function.
        **kwargs:
            Any additional algorithm-specific instantiation keyword argument.

        Returns
        -------
        controller:
            `FairnessControllerServer` instance matching input specifications.

        Raises
        ------
        KeyError
            If `algorithm` does not match any registered
            `FairnessControllerServer` type.
        """
        try:
            cls = access_registered(
                name=algorithm, group="FairnessControllerServer"
            )
        except Exception as exc:
            raise KeyError(
                "Failed to retrieve fairness controller with algorithm name "
                f"'{algorithm}'."
            ) from exc
        assert issubclass(cls, FairnessControllerServer)
        return cls(f_type=f_type, f_args=f_args, **kwargs)
