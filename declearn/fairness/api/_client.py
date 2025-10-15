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

"""Client-side ABC for fairness-aware federated learning controllers."""

import abc
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np

from declearn.communication.api import NetworkClient
from declearn.communication.utils import verify_server_message_validity
from declearn.fairness.api._dataset import FairnessDataset
from declearn.fairness.api._fair_func import instantiate_fairness_function
from declearn.fairness.api._metrics import FairnessMetricsComputer
from declearn.messaging import (
    Error,
    FairnessCounts,
    FairnessGroups,
    FairnessQuery,
    FairnessReply,
    FairnessSetupQuery,
)
from declearn.metrics import MeanMetric
from declearn.secagg.api import Encrypter
from declearn.secagg.messaging import (
    SecaggFairnessCounts,
    SecaggFairnessReply,
)
from declearn.training import TrainingManager
from declearn.utils import (
    access_registered,
    create_types_registry,
    register_type,
)

__all__ = [
    "FairnessControllerClient",
]


@create_types_registry(name="FairnessControllerClient")
class FairnessControllerClient(metaclass=abc.ABCMeta):
    """Abstract base class for client-side fairness controllers.

    Usage
    -----
    A `FairnessControllerClient` (subclass) instance has two main
    routines that are to be called as part of a federated learning
    process, in addition to a static method from the base API class:

    - `from_setup_query`:
        This is a static method that can be called generically from
        the base `FairnessControllerClient` type to instantiate a
        controller from a server-emitted `FairnessSetupQuery`.
    - `setup_fairness`:
        This routine is to be called only once, after instantiating
        from a `FairnessSetupQuery`. It triggers the following process:
            - Run a basic routine to exchange sensitive group definitions
              and associated (encrypted) sample counts.
            - Perform any additional algorithm-specific setup actions.
    - `run_fairness_round`:
        This routine is to be called once per round, before the next
        training round occurs, upon receiving a `FairnessQuery` from
        the server. It triggers the following process:
            - Run a basic routine to compute fairness-related metrics
              and send (some of) their (encrypted) values to the server.
            - Perform any additonal algorithm-specific round actions.

    Inheritance
    -----------
    Algorithm-specific subclasses should define the following abstract
    attribute and methods:

    - `algorithm`:
        Abstract string class attribute. Name under which this controller
        and its server-side counterpart classes are registered.
    - `finalize_fairness_setup`:
        Method implementing any algorithm-specific setup actions.
    - `finalize_fairness_round`:
        Method implementing any algorithm-specific round actions.

    Additionally, they may overload or override the following method:

    - `setup_fairness_metrics`:
        Method that defines metrics being computed as part of fairness
        rounds. By default, group-wise accuracy values are computed and
        shared with the server, and the local fairness is computed from
        them (but not sent to the server).

    By default, subclasses are type-registered under their `algorithm`
    name and "FairnessControllerClient" group upon declaration. This can
    be prevented by passing `register=False` to the inheritance parameters
    (e.g. `class Cls(FairnessControllerClient, register=False)`).
    See `declearn.utils.register_type` for details on types registration.
    """

    algorithm: ClassVar[str]
    """Name of the fairness-enforcing algorithm.

    This name should be unique across 'FairnessControllerClient' classes,
    and shared with a unique paired 'FairnessControllerServer'. It is used
    for type-registration and to enable instantiating a client controller
    based on server-emitted instructions in a federated setting.
    """

    def __init_subclass__(
        cls,
        register: bool = True,
    ) -> None:
        """Automatically type-register subclasses."""
        if register:
            register_type(cls, cls.algorithm, group="FairnessControllerClient")

    def __init__(
        self,
        manager: TrainingManager,
        f_type: str,
        f_args: Dict[str, Any],
    ) -> None:
        """Instantiate the client-side fairness controller.

        Parameters
        ----------
        manager:
            `TrainingManager` instance wrapping the model being trained
            and its training dataset (that must be a `FairnessDataset`).
        f_type:
            Name of the type of group-fairness function being optimized.
        f_args:
            Keyword arguments to the group-fairness function.
        """
        if not isinstance(manager.train_data, FairnessDataset):
            raise TypeError(
                "Cannot set up fairness without a 'FairnessDataset' "
                "as training dataset."
            )
        self.manager = manager
        self.computer = FairnessMetricsComputer(manager.train_data)
        self.fairness_function = instantiate_fairness_function(
            f_type=f_type, counts=self.computer.counts, **f_args
        )
        self.groups: List[Tuple[Any, ...]] = []

    @staticmethod
    def from_setup_query(
        query: FairnessSetupQuery,
        manager: TrainingManager,
    ) -> "FairnessControllerClient":
        """Instantiate a controller from a server-emitted query.

        Parameters
        ----------
        query:
            `FairnessSetupQuery` received from the server.
        manager:
            `TrainingManager` wrapping the model to train.

        Returns
        -------
        controller:
            `FairnessControllerClient` instance, the type and parameters
            of which depend on the input `query`, that wraps `manager`.
        """
        try:
            cls = access_registered(
                name=query.algorithm, group="FairnessControllerClient"
            )
            assert issubclass(cls, FairnessControllerClient)
        except Exception as exc:
            raise ValueError(
                "Failed to retrieve a 'FairnessControllerClient' class "
                "matching the input 'FairnessSetupQuery' message."
            ) from exc
        return cls(manager=manager, **query.params)

    async def setup_fairness(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
    ) -> None:
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
        secagg:
            Optional SecAgg encryption controller.
        """
        # Agree on a list of sensitive groups and share local sample counts.
        await self.exchange_sensitive_groups_list_and_counts(netwk, secagg)
        # Run additional algorithm-specific setup steps.
        await self.finalize_fairness_setup(netwk, secagg)

    async def exchange_sensitive_groups_list_and_counts(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
    ) -> None:
        """Agree on a list of sensitive groups and share local sample counts.

        This method performs the following routine:

        - Send the list of local sensitive group definitions to the server.
        - Await a unified list of sensitive groups in return.
        - Assign the received list as `groups` attribute.
        - Send (optionally-encrypted) group-wise sample counts to the server.

        Parameters
        ----------
        netwk:
            `NetworkClient` endpoint, connected to a server.
        secagg:
            Optional SecAgg encryption controller.
        """
        # Share sensitive groups definitions and received an ordered list.
        self.groups = await self._exchange_sensitive_groups_list(netwk)
        # Send group-wise sample counts for the server to (secure-)aggregate.
        await self._send_sensitive_groups_counts(netwk, secagg)

    async def _exchange_sensitive_groups_list(
        self,
        netwk: NetworkClient,
    ) -> List[Tuple[Any, ...]]:
        """Exhange sensitive groups definitions and return a unified list."""
        # Gather local sensitive groups and their sample counts.
        counts = self.computer.counts
        groups = list(counts)
        # Share them and receive a unified, ordered list of groups.
        await netwk.send_message(FairnessGroups(groups=groups))
        received = await netwk.recv_message()
        message = await verify_server_message_validity(
            netwk, received, expected=FairnessGroups
        )
        return message.groups

    async def _send_sensitive_groups_counts(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
    ) -> None:
        """Send (opt. encrypted) group-wise sample counts to the server."""
        counts = self.computer.counts
        reply = FairnessCounts([counts.get(group, 0) for group in self.groups])
        if secagg is None:
            await netwk.send_message(reply)
        else:
            await netwk.send_message(
                SecaggFairnessCounts.from_cleartext_message(reply, secagg)
            )

    @abc.abstractmethod
    async def finalize_fairness_setup(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
    ) -> None:
        """Finalize the fairness setup routine.

        This method is called as part of `setup_fairness`, and should
        be defined by concrete subclasses to implement setup behavior
        once the initial echange of sensitive group definitions and
        sample counts has been performed.

        Parameters
        ----------
        netwk:
            NetworkClient endpoint, registered to a server.
        secagg:
            Optional SecAgg encryption controller.
        """

    async def run_fairness_round(
        self,
        netwk: NetworkClient,
        query: FairnessQuery,
        secagg: Optional[Encrypter],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Participate in a round of actions to enforce fairness.

        Parameters
        ----------
        netwk:
            NetworkClient endpoint instance, connected to a server.
        query:
            `FairnessQuery` message to participate in a fairness round.
        secagg:
            Optional SecAgg encryption controller.

        Returns
        -------
        metrics:
            Fairness(-related) metrics computed as part of this routine,
            as a `{name: value}` dict with scalar or numpy array values.
        """
        try:
            values = await self._compute_and_share_fairness_measures(
                netwk, query, secagg
            )
        except Exception as exc:
            error = f"Error encountered in fairness round: {repr(exc)}"
            self.manager.logger.error(error)
            await netwk.send_message(Error(error))
            raise RuntimeError(error) from exc
        # Run additional algorithm-specific steps.
        return await self.finalize_fairness_round(netwk, secagg, values)

    async def _compute_and_share_fairness_measures(
        self,
        netwk: NetworkClient,
        query: FairnessQuery,
        secagg: Optional[Encrypter],
    ) -> Dict[str, Dict[Tuple[Any, ...], float]]:
        """Compute, share (encrypted) and return fairness measures."""
        # Optionally update the wrapped model's weights.
        if query.weights is not None:
            self.manager.model.set_weights(query.weights, trainable=True)
        # Compute some fairness-related values, split between two sets.
        share_values, local_values = self.compute_fairness_measures(
            query.batch_size, query.n_batch, query.thresh
        )
        # Share the first set of values for their (secure-)aggregation.
        reply = FairnessReply(values=share_values)
        if secagg is None:
            await netwk.send_message(reply)
        else:
            await netwk.send_message(
                SecaggFairnessReply.from_cleartext_message(reply, secagg)
            )
        # Return the second set of values.
        return local_values

    def compute_fairness_measures(
        self,
        batch_size: int,
        n_batch: Optional[int] = None,
        thresh: Optional[float] = None,
    ) -> Tuple[List[float], Dict[str, Dict[Tuple[Any, ...], float]]]:
        """Compute fairness measures based on a received query.

        By default, compute and return group-wise accuracy metrics,
        weighted by group-wise sample counts. This may be modified
        by algorithm-specific subclasses depending on algorithms'
        needs.

        Parameters
        ----------
        batch_size:
            Number of samples per batch when computing predictions.
        n_batch:
            Optional maximum number of batches to draw per category.
            If None, use the entire wrapped dataset.
        thresh:
            Optional binarization threshold for binary classification
            models' output scores. If None, use 0.5 by default, or 0.0
            for `SklearnSGDModel` instances.
            Unused for multinomial classifiers (argmax over scores).

        Returns
        -------
        share_values:
            Values that are to be shared with the orchestrating server,
            as a deterministic-length list of float values.
        local_values:
            Values that are to be used in local post-processing steps,
            as a nested dictionary of group-wise metrics.
        """
        # Compute group-wise metrics.
        metrics = self.setup_fairness_metrics(thresh=thresh)
        local_values = self.computer.compute_groupwise_metrics(
            metrics=metrics,
            model=self.manager.model,
            batch_size=batch_size,
            n_batch=n_batch,
        )
        # Gather sample-counts-scaled values to share with the server.
        scaled_values = {
            key: self.computer.scale_metrics_by_sample_counts(val)
            for key, val in local_values.items()
        }
        share_values = [
            scaled_values[key].get(group, 0.0)
            for key in sorted(scaled_values)
            for group in self.groups
        ]
        # Compute group-wise local fairness measures.
        if "accuracy" in local_values:
            fairness = self.fairness_function.compute_from_group_accuracy(
                local_values["accuracy"]
            )
            local_values[self.fairness_function.f_type] = fairness
        # Return both shareable and local values.
        return share_values, local_values

    def setup_fairness_metrics(
        self,
        thresh: Optional[float] = None,
    ) -> List[MeanMetric]:
        """Setup metrics to compute group-wise and share with the server.

        By default, this method returns an accuracy-computation method.
        It may be overloaded to compute additional metrics depending on
        the needs of the fairness-enforcing algorithm being implemented.

        Parameters
        ----------
        thresh:
            Optional binarization threshold for binary classification
            models' output scores. Used to setup accuracy computations.

        Returns
        -------
        metrics:
            List of `MeanMetric` instances, that each compute a unique
            scalar float metric (per sensitive group) and have distinct
            names.
        """
        accuracy = self.computer.setup_accuracy_metric(
            self.manager.model, thresh=thresh
        )
        return [accuracy]

    @abc.abstractmethod
    async def finalize_fairness_round(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
        values: Dict[str, Dict[Tuple[Any, ...], float]],
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Take actions to enforce fairness.

        This method is designed to be called after an initial query
        has been received and responded to, resulting in computing
        and sharing fairness(-related) metrics.

        Parameters
        ----------
        netwk:
            NetworkClient endpoint instance, connected to a server.
        secagg:
            Optional SecAgg encryption controller.
        values:
            Nested dict of locally-computed group-wise metrics.
            This is the second set of `compute_fairness_measures` return
            values; when this method is called, the first has already
            been shared with the server for (secure-)aggregation.

        Returns
        -------
        metrics:
            Computed fairness(-related) metrics to checkpoint, as a
            `{name: value}` dict with scalar or numpy array values.
        """
