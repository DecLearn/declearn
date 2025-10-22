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

"""Client-side Fed-FairBatch/FedFB controller."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from declearn.aggregator import SumAggregator
from declearn.communication.api import NetworkClient
from declearn.communication.utils import verify_server_message_validity
from declearn.fairness.api import (
    FairnessControllerClient,
    FairnessDataset,
)
from declearn.fairness.fairbatch._dataset import FairbatchDataset
from declearn.fairness.fairbatch._messages import (
    FairbatchOkay,
    FairbatchSamplingProbas,
)
from declearn.messaging import Error
from declearn.metrics import MeanMetric
from declearn.secagg.api import Encrypter
from declearn.training import TrainingManager

__all__ = [
    "FairbatchControllerClient",
]


class FairbatchControllerClient(FairnessControllerClient):
    """Client-side controller to implement Fed-FairBatch or FedFB."""

    algorithm = "fairbatch"

    def __init__(
        self,
        manager: TrainingManager,
        f_type: str,
        f_args: Dict[str, Any],
    ) -> None:
        super().__init__(manager=manager, f_type=f_type, f_args=f_args)
        assert isinstance(self.manager.train_data, FairnessDataset)
        self.manager.train_data = FairbatchDataset(self.manager.train_data)

    async def finalize_fairness_setup(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
    ) -> None:
        # Force the use of a SumAggregator.
        if not isinstance(self.manager.aggrg, SumAggregator):
            self.manager.aggrg = SumAggregator()
        # Receive and assign initial sampling probabilities.
        await self._update_fairbatch_sampling_probas(netwk)

    async def _update_fairbatch_sampling_probas(
        self,
        netwk: NetworkClient,
    ) -> None:
        """Run a FairBatch-specific routine to update sampling probabilities.

        Expect a message from the orchestrating server containing the new
        sensitive group sampling probabilities, and apply them to the
        training dataset.

        Raises
        ------
        RuntimeError:
            If the expected message is not received.
            If the sampling pobabilities' update fails.
        """
        # Receive aggregated sensitive weights.
        received = await netwk.recv_message()
        message = await verify_server_message_validity(
            netwk, received, expected=FairbatchSamplingProbas
        )
        probas = dict(zip(self.groups, message.probas, strict=False))
        # Set the received weights, handling and propagating exceptions if any.
        try:
            assert isinstance(self.manager.train_data, FairbatchDataset)
            self.manager.train_data.set_sampling_probabilities(
                group_probas=probas
            )
        except Exception as exc:
            self.manager.logger.error(
                "Exception encountered when setting FairBatch sampling"
                "probabilities: %s",
                repr(exc),
            )
            await netwk.send_message(Error(repr(exc)))
            raise RuntimeError(
                "FairBatch sampling probabilities update failed."
            ) from exc
        # If things went well, ping the server back to indicate so.
        self.manager.logger.info("Updated FairBatch sampling probabilities.")
        await netwk.send_message(FairbatchOkay())

    def setup_fairness_metrics(
        self,
        thresh: Optional[float] = None,
    ) -> List[MeanMetric]:
        loss = self.computer.setup_loss_metric(model=self.manager.model)
        metrics = super().setup_fairness_metrics(thresh=thresh)
        metrics.append(loss)
        return metrics

    async def finalize_fairness_round(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
        values: Dict[str, Dict[Tuple[Any, ...], float]],
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Await updated loss weights from the server.
        await self._update_fairbatch_sampling_probas(netwk)
        # Return group-wise local accuracy, model loss and fairness scores.
        return {
            f"{metric}_{group}": value
            for metric, m_dict in values.items()
            for group, value in m_dict.items()
        }
