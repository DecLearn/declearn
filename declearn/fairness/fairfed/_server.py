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

"""Server-side FairFed controller."""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from declearn.aggregator import Aggregator
from declearn.communication.api import NetworkServer
from declearn.communication.utils import verify_client_messages_validity
from declearn.fairness.api import (
    FairnessControllerServer,
    instantiate_fairness_function,
)
from declearn.fairness.fairfed._aggregator import FairfedAggregator
from declearn.fairness.fairfed._fairfed import FairfedValueComputer
from declearn.fairness.fairfed._messages import (
    FairfedDelta,
    FairfedDeltavg,
    FairfedFairness,
    FairfedOkay,
    SecaggFairfedDelta,
)
from declearn.messaging import FairnessSetupQuery
from declearn.secagg.api import Decrypter
from declearn.secagg.messaging import aggregate_secagg_messages

__all__ = [
    "FairfedControllerServer",
]


class FairfedControllerServer(FairnessControllerServer):
    """Server-side controller to implement FairFed.

    FairFed [1] is an algorithm that aims at enforcing fairness in
    a federated learning setting by altering the aggregation rule
    for client-wise model updates. It conducts a weighted averaging
    of these updates that is based on discrepancy metrics between
    global and client-wise fairness measures.

    This algorithm was originally designed for settings where a binary
    classifier is trained over data with a single binary sensitive
    attribute, with the authors showcasing their generic formulas over
    a limited set of group fairness definitions. DecLearn expands it to
    a broader case, enabling the use of arbitrary fairness definitions
    over data that may have non-binary and/or many sensitive attributes.
    A 'strict' mode is made available to stick to the original paper,
    that is turned on by default and can be disabled at instantiation.

    It is worth noting that the authors of FairFed suggest combining it
    with other mechanisms that aim at enforcing local model fairness; at
    the moment, this is not implemented in DecLearn, unless a custom and
    specific `Model` subclass is implemented by end-users to do so.

    References
    ----------
    - [1]
        Ezzeldin et al. (2021).
        FairFed: Enabling Group Fairness in Federated Learning
        https://arxiv.org/abs/2110.00857
    """

    algorithm = "fairfed"

    # pylint: disable-next=too-many-positional-arguments
    def __init__(
        self,
        f_type: str,
        f_args: Optional[Dict[str, Any]] = None,
        beta: float = 1.0,
        strict: bool = True,
        target: Optional[int] = None,
    ) -> None:
        """Instantiate the server-side Fed-FairGrad controller.

        Parameters
        ----------
        f_type:
            Name of the fairness function to evaluate and optimize.
        f_args:
            Optional dict of keyword arguments to the fairness function.
        beta:
            Hyper-parameter controlling the magnitude of updates
            to clients' averaging weights updates.
        strict:
            Whether to stick strictly to the FairFed paper's setting
            and explicit formulas, or to use a broader adaptation of
            FairFed to more diverse settings.
        target:
            If `strict=True`, target value of interest, on which to focus.
            If None, try fetching from `f_args` or use default value `1`.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(f_type=f_type, f_args=f_args)
        self.beta = beta
        # Set up a temporary fairness function, replaced at setup time.
        self._fairness = instantiate_fairness_function(
            "accuracy_parity", counts={}
        )
        # Set up an uninitialized FairFed value computer.
        if target is None:
            target = int(self.f_args.get("target", 1))
        self.fairfed_computer = FairfedValueComputer(
            f_type=self.f_type, strict=strict, target=target
        )

    @property
    def strict(
        self,
    ) -> bool:
        """Whether this controller strictly sticks to the FairFed paper."""
        return self.fairfed_computer.strict

    def prepare_fairness_setup_query(
        self,
    ) -> FairnessSetupQuery:
        query = super().prepare_fairness_setup_query()
        query.params["beta"] = self.beta
        query.params["strict"] = self.strict
        query.params["target"] = self.fairfed_computer.target
        return query

    async def finalize_fairness_setup(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
        counts: List[int],
        aggregator: Aggregator,
    ) -> Aggregator:
        # Set up a fairness function and initialized the FairFed computer.
        self._fairness = instantiate_fairness_function(
            self.f_type,
            counts=dict(zip(self.groups, counts, strict=False)),
            **self.f_args,
        )
        self.fairfed_computer.initialize(groups=self.groups)
        # Force the use of a FairFed-specific averaging aggregator.
        warnings.warn(
            "Overriding Aggregator choice due to the use of FairFed.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return FairfedAggregator(beta=self.beta)

    async def finalize_fairness_round(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
        values: List[float],
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Unpack group-wise accuracy values and compute fairness ones.
        accuracy = dict(zip(self.groups, values, strict=False))
        fairness = self._fairness.compute_from_federated_group_accuracy(
            accuracy
        )
        # Share the absolute mean fairness with clients.
        fair_avg = self.fairfed_computer.compute_synthetic_fairness_value(
            fairness
        )
        await netwk.broadcast_message(FairfedFairness(fairness=fair_avg))
        # Await and (secure-)aggregate clients' absolute fairness difference.
        received = await netwk.wait_for_messages()
        if secagg is None:
            replies = await verify_client_messages_validity(
                netwk, received, expected=FairfedDelta
            )
            deltavg = sum(r.delta for r in replies.values()) / len(replies)
        else:
            sec_rep = await verify_client_messages_validity(
                netwk, received, expected=SecaggFairfedDelta
            )
            deltavg = aggregate_secagg_messages(sec_rep, secagg).delta
        # Share the computed value with clients and await their okay signal.
        await netwk.broadcast_message(FairfedDeltavg(deltavg=deltavg))
        received = await netwk.wait_for_messages()
        await verify_client_messages_validity(
            netwk, received, expected=FairfedOkay
        )
        # Package and return accuracy, fairness and computed average metrics.
        metrics: Dict[str, Union[float, np.ndarray]] = {
            f"accuracy_{key}": val for key, val in accuracy.items()
        }
        metrics.update(
            {f"{self.f_type}_{key}": val for key, val in fairness.items()}
        )
        metrics["fairfed_value"] = fair_avg
        metrics["fairfed_deltavg"] = deltavg
        return metrics
