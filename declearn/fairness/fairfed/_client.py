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

"""Client-side FairFed controller."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from declearn.communication.api import NetworkClient
from declearn.communication.utils import verify_server_message_validity
from declearn.fairness.api import FairnessControllerClient
from declearn.fairness.fairfed._aggregator import FairfedAggregator
from declearn.fairness.fairfed._fairfed import FairfedValueComputer
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

    # pylint: disable-next=too-many-positional-arguments
    def __init__(  # noqa: PLR0913
        self,
        manager: TrainingManager,
        f_type: str,
        f_args: Dict[str, Any],
        beta: float,
        strict: bool = True,
        target: int = 1,
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
        target:
            Choice of target label to focus on in `strict` mode.
            Unused when `strict=False`.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(manager=manager, f_type=f_type, f_args=f_args)
        self.beta = beta
        self.fairfed_computer = FairfedValueComputer(
            f_type=self.fairness_function.f_type, strict=strict, target=target
        )
        self.fairfed_computer.initialize(groups=self.fairness_function.groups)

    @property
    def strict(
        self,
    ) -> bool:
        """Whether this function strictly sticks to the FairFed paper."""
        return self.fairfed_computer.strict

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

    async def finalize_fairness_round(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
        values: Dict[str, Dict[Tuple[Any, ...], float]],
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Await absolute mean fairness across all clients.
        received = await netwk.recv_message()
        fair_glb = await verify_server_message_validity(
            netwk, received, expected=FairfedFairness
        )
        # Compute the absolute difference between local and global fairness.
        fair_avg = self.fairfed_computer.compute_synthetic_fairness_value(
            values[self.fairness_function.f_type]
        )
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
        # Flatten group-wise local accuracy and fairness scores.
        metrics: Dict[str, Union[float, np.ndarray]] = {
            f"{metric}_{group}": value
            for metric, m_dict in values.items()
            for group, value in m_dict.items()
        }
        # Add FairFed-specific metrics, then return.
        metrics["fairfed_value"] = fair_avg
        metrics["fairfed_delta"] = my_delta.delta
        metrics["fairfed_deltavg"] = deltavg.deltavg
        return metrics
