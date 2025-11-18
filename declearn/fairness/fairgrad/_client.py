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

"""Client-side Fed-FairGrad controller."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from declearn.aggregator import SumAggregator
from declearn.communication.api import NetworkClient
from declearn.communication.utils import verify_server_message_validity
from declearn.fairness.api import (
    FairnessControllerClient,
    FairnessDataset,
)
from declearn.fairness.fairgrad._messages import FairgradOkay, FairgradWeights
from declearn.messaging import Error
from declearn.secagg.api import Encrypter

__all__ = [
    "FairgradControllerClient",
]


class FairgradControllerClient(FairnessControllerClient):
    """Client-side controller to implement Fed-FairGrad."""

    algorithm = "fairgrad"

    async def finalize_fairness_setup(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
    ) -> None:
        # Force the use of a SumAggregator.
        if not isinstance(self.manager.aggrg, SumAggregator):
            self.manager.aggrg = SumAggregator()
        # Await initial loss weights from the server.
        await self._update_fairgrad_weights(netwk)

    async def _update_fairgrad_weights(
        self,
        netwk: NetworkClient,
    ) -> None:
        """Run a FairGrad-specific routine to update sensitive group weights.

        Expect a message from the orchestrating server containing the new
        sensitive group weights, and apply them to the training dataset.

        Raises
        ------
        RuntimeError:
            If the expected message is not received.
            If the weights' update fails.
        """
        # Receive aggregated sensitive weights.
        received = await netwk.recv_message()
        message = await verify_server_message_validity(
            netwk, received, expected=FairgradWeights
        )
        weights = dict(zip(self.groups, message.weights, strict=False))
        # Set the received weights, handling and propagating exceptions if any.
        try:
            assert isinstance(self.manager.train_data, FairnessDataset)
            self.manager.train_data.set_sensitive_group_weights(
                weights, adjust_by_counts=True
            )
        except Exception as exc:
            self.manager.logger.error(
                "Exception encountered when setting FairGrad weights: %s", exc
            )
            await netwk.send_message(Error(repr(exc)))
            raise RuntimeError("FairGrad weights update failed.") from exc
        # If things went well, ping the server back to indicate so.
        self.manager.logger.info("Updated FairGrad weights.")
        await netwk.send_message(FairgradOkay())

    async def finalize_fairness_round(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
        values: Dict[str, Dict[Tuple[Any, ...], float]],
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Await updated loss weights from the server.
        await self._update_fairgrad_weights(netwk)
        # Return group-wise local accuracy and fairness scores.
        return {
            f"{metric}_{group}": value
            for metric, m_dict in values.items()
            for group, value in m_dict.items()
        }
