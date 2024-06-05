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

"""Server-side FairFed controller."""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from declearn.aggregator import Aggregator
from declearn.communication.api import NetworkServer
from declearn.communication.utils import verify_client_messages_validity
from declearn.fairness.api import FairnessControllerServer
from declearn.fairness.core import instantiate_fairness_function
from declearn.fairness.fairfed._aggregator import FairfedAggregator
from declearn.fairness.fairfed._function import FairfedFairnessFunction
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
    """Server-side controller to implement FairFed."""

    algorithm = "fairfed"

    def __init__(
        self,
        f_type: str,
        f_args: Optional[Dict[str, Any]] = None,
        beta: float = 1.0,
        strict: bool = True,
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
        """
        super().__init__(f_type=f_type, f_args=f_args)
        self.beta = beta
        # Set up a temporary fairness function, replaced at setup time.
        fairfed_func = instantiate_fairness_function(
            "accuracy_parity", counts={}
        )
        self.fairfed_func = FairfedFairnessFunction(
            wrapped=fairfed_func, strict=strict
        )

    @property
    def strict(
        self,
    ) -> bool:
        """Whether this controller strictly sticks to the FairFed paper."""
        return self.fairfed_func.strict

    def prepare_fairness_setup_query(
        self,
    ) -> FairnessSetupQuery:
        query = super().prepare_fairness_setup_query()
        query.params["beta"] = self.beta
        query.params["strict"] = self.strict
        return query

    async def finalize_fairness_setup(
        self,
        netwk: NetworkServer,
        counts: List[int],
        aggregator: Aggregator,
    ) -> Aggregator:
        # Set up a fairness function.
        fairfed_func = instantiate_fairness_function(
            self.f_type, counts=dict(zip(self.groups, counts)), **self.f_args
        )
        self.fairfed_func = FairfedFairnessFunction(
            wrapped=fairfed_func, strict=self.fairfed_func.strict
        )
        # Force the use of a FairFed-specific averaging aggregator.
        warnings.warn(
            "Overriding Aggregator choice due to the use of FairFed.",
            category=RuntimeWarning,
        )
        return FairfedAggregator(beta=self.beta)

    async def finalize_fairness_round(
        self,
        round_i: int,
        values: List[float],
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Unpack group-wise accuracy values and compute fairness.
        accuracy = dict(zip(self.groups, values))
        fairness = self.fairfed_func.compute_group_fairness_from_accuracy(
            accuracy, federated=True
        )
        # Share the absolute mean fairness with clients.
        fair_avg = self.fairfed_func.compute_synthetic_fairness_value(fairness)
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
        metrics = {
            f"accuracy_{key}": val for key, val in accuracy.items()
        }  # type: Dict[str, Union[float, np.ndarray]]
        metrics.update(
            {f"{self.f_type}_{key}": val for key, val in fairness.items()}
        )
        metrics[f"{self.f_type}_mean_abs"] = fair_avg
        metrics["fairfed_deltavg"] = deltavg
        return metrics
