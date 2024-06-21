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

"""Client-side FairFed controller."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from declearn.communication.api import NetworkClient
from declearn.communication.utils import verify_server_message_validity
from declearn.fairness.api import FairnessControllerClient
from declearn.fairness.fairfed._aggregator import FairfedAggregator
from declearn.fairness.fairfed._function import FairfedFairnessFunction
from declearn.fairness.fairfed._messages import (
    FairfedDelta,
    FairfedDeltavg,
    FairfedFairness,
    FairfedOkay,
    SecaggFairfedDelta,
)
from declearn.secagg.api import Encrypter
from declearn.training import TrainingManager

__all__ = [
    "FairfedControllerClient",
]


class FairfedControllerClient(FairnessControllerClient):
    """Client-side controller to implement FairFed."""

    algorithm = "fairfed"

    def __init__(
        self,
        manager: TrainingManager,
        f_type: str,
        f_args: Dict[str, Any],
        beta: float,
        strict: bool = True,
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
        beta:
            Hyper-parameter controlling the magnitude of averaging weights'
            updates across rounds.
        strict:
            Whether to stick strictly to the FairFed paper's setting
            and explicit formulas, or to use a broader adaptation of
            FairFed to more diverse settings.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(manager=manager, f_type=f_type, f_args=f_args)
        self.beta = beta
        self._key_groups = (
            ((0, 0), (0, 1)) if strict else None
        )  # type: Optional[Tuple[Tuple[Any, ...], Tuple[Any, ...]]]
        self.fairfed_func = FairfedFairnessFunction(
            self.fairness_function, strict=strict
        )

    @property
    def strict(
        self,
    ) -> bool:
        """Whether this controller strictly sticks to the FairFed paper."""
        return self.fairfed_func.strict

    async def finalize_fairness_setup(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
    ) -> None:
        # Force the use of a FairFed-specific aggregator.
        self.manager.aggrg = FairfedAggregator(beta=self.beta)
        self.manager.aggrg.initialize_local_weight(
            n_samples=sum(self.computer.counts.values())
        )

    def compute_fairness_measures(
        self,
        batch_size: int,
        n_batch: Optional[int] = None,
        thresh: Optional[float] = None,
    ) -> Tuple[List[float], List[float]]:
        # Compute group-wise accuracy and fairness scores.
        # pylint: disable=duplicate-code
        accuracy = self.computer.compute_groupwise_accuracy(
            model=self.manager.model,
            batch_size=batch_size,
            n_batch=n_batch,
            thresh=thresh,
        )
        fairness = self.fairfed_func.compute_group_fairness_from_accuracy(
            accuracy, federated=False
        )
        # pylint: enable=duplicate-code
        # Flatten local values for post-processing and checkpointing.
        local_values = list(accuracy.values()) + list(fairness.values())
        # Scale accuracy values by sample counts for their aggregation.
        accuracy = self.computer.scale_metrics_by_sample_counts(accuracy)
        # Flatten shareable values, ordered and filled-out.
        share_values = [accuracy.get(group, 0.0) for group in self.groups]
        # Return both sets of values.
        return share_values, local_values

    async def finalize_fairness_round(
        self,
        netwk: NetworkClient,
        values: List[float],
        secagg: Optional[Encrypter],
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Recover local accuracy and fairness values.
        groups = list(self.computer.g_data)
        accuracy = dict(zip(groups, values[: len(groups)]))
        fairness = dict(zip(groups, values[len(groups) :]))
        # Await absolute mean fairness across all clients.
        received = await netwk.recv_message()
        fair_glb = await verify_server_message_validity(
            netwk, received, expected=FairfedFairness
        )
        # Compute the absolute difference between local and global fairness.
        fair_avg = self.fairfed_func.compute_synthetic_fairness_value(fairness)
        my_delta = FairfedDelta(abs(fair_avg - fair_glb.fairness))
        # Share it with the server for its (secure-)aggregation across clients.
        if secagg is None:
            await netwk.send_message(my_delta)
        else:
            await netwk.send_message(
                SecaggFairfedDelta.from_cleartext_message(my_delta, secagg)
            )
        # Await mean absolute fairness difference across clients.
        received = await netwk.recv_message()
        deltavg = await verify_server_message_validity(
            netwk, received, expected=FairfedDeltavg
        )
        # Update the aggregation weight of this client.
        assert isinstance(self.manager.aggrg, FairfedAggregator)
        self.manager.aggrg.update_local_weight(
            delta_loc=my_delta.delta,
            delta_avg=deltavg.deltavg,
        )
        # Signal the server that things went well.
        await netwk.send_message(FairfedOkay())
        # Package and return accuracy, fairness and fairfed metrics.
        metrics = {
            f"accuracy_{key}": val for key, val in accuracy.items()
        }  # type: Dict[str, Union[float, np.ndarray]]
        f_type = self.fairfed_func.f_type
        metrics.update(
            {f"{f_type}_{key}": val for key, val in fairness.items()}
        )
        metrics[f"{f_type}_mean_abs"] = fair_avg
        metrics["fairfed_delta"] = my_delta.delta
        metrics["fairfed_deltavg"] = deltavg.deltavg
        return metrics
