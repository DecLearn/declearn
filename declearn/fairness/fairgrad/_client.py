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

"""Client-side Fed-FairGrad controller."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from declearn.communication.api import NetworkClient
from declearn.communication.utils import verify_server_message_validity
from declearn.fairness.api import FairnessControllerClient
from declearn.fairness.core import (
    FairnessDataset,
    instantiate_fairness_function,
)
from declearn.fairness.fairgrad._messages import FairgradOkay, FairgradWeights
from declearn.messaging import Error
from declearn.secagg.api import Encrypter
from declearn.training import TrainingManager

__all__ = [
    "FairgradControllerClient",
]


class FairgradControllerClient(FairnessControllerClient):
    """Client-side controller to implement Fed-FairGrad."""

    algorithm = "fedfairgrad"

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
        super().__init__(manager)
        self.fairness_function = instantiate_fairness_function(
            f_type=f_type, counts=self.computer.counts, **f_args
        )

    async def finalize_fairness_setup(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
    ) -> None:
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
        received = await netwk.check_message()
        message = await verify_server_message_validity(
            netwk, received, expected=FairgradWeights
        )
        weights = dict(zip(self.groups, message.weights))
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

    def compute_fairness_measures(
        self,
        batch_size: int,
        n_batch: Optional[int] = None,
        thresh: Optional[float] = None,
    ) -> Tuple[List[float], List[float]]:
        # Compute group-wise accuracy scores.
        accuracy = self.computer.compute_groupwise_accuracy(
            model=self.manager.model,
            batch_size=batch_size,
            n_batch=n_batch,
            thresh=thresh,
        )
        # Flatten local values for post-processing and checkpointing.
        local_values = list(accuracy.values())
        # Scale local values by sample counts for their aggregation.
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
        # Await updated loss weights from the server.
        await self._update_fairgrad_weights(netwk)
        # Recover raw accuracy scores for groups with local samples.
        accuracy = dict(zip(self.computer.g_data, values))
        # Compute local fairness measures.
        fairness = self.fairness_function.compute_from_group_accuracy(accuracy)
        f_type = self.fairness_function.f_type
        # Package and return accuracy and fairness metrics.
        metrics = {
            f"accuracy_{key}": val for key, val in accuracy.items()
        }  # type: Dict[str, Union[float, np.ndarray]]
        metrics.update(
            {f"{f_type}_{key}": val for key, val in fairness.items()}
        )
        return metrics
