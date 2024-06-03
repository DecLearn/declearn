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

"""Client-side Fed-FairBatch controller."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from declearn.communication.api import NetworkClient
from declearn.communication.utils import verify_server_message_validity
from declearn.fairness.api import FairnessControllerClient
from declearn.fairness.core import (
    FairnessDataset,
    instantiate_fairness_function,
)
from declearn.fairness.fairbatch._dataset import FairbatchDataset
from declearn.fairness.fairbatch._messages import (
    FairbatchSamplingProbas,
    FairbatchOkay,
)
from declearn.messaging import Error
from declearn.secagg.api import Encrypter
from declearn.training import TrainingManager

__all__ = [
    "FairbatchControllerClient",
]


class FairbatchControllerClient(FairnessControllerClient):
    """Client-side controller to implement Fed-FairBatch or FedFB."""

    algorithm = "fedfairbatch"

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
        assert isinstance(self.manager.train_data, FairnessDataset)
        self.manager.train_data = FairbatchDataset(self.manager.train_data)
        self.fairness_function = instantiate_fairness_function(
            f_type=f_type, counts=self.computer.counts, **f_args
        )

    async def finalize_fairness_setup(
        self,
        netwk: NetworkClient,
        secagg: Optional[Encrypter],
    ) -> None:
        pass  # no action required beyond sharing group definitions and counts

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
        received = await netwk.check_message()
        message = await verify_server_message_validity(
            netwk, received, expected=FairbatchSamplingProbas
        )
        probas = dict(zip(self.groups, message.probas))
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

    def compute_fairness_measures(
        self,
        batch_size: int,
        n_batch: Optional[int] = None,
        thresh: Optional[float] = None,
    ) -> Tuple[List[float], List[float]]:
        # Compute group-wise accuracy scores and loss values.
        accuracy, loss = self.computer.compute_groupwise_accuracy_and_loss(
            model=self.manager.model,
            batch_size=batch_size,
            n_batch=n_batch,
            thresh=thresh,
        )
        # Flatten local values for post-processing and checkpointing.
        local_values = list(accuracy.values()) + list(loss.values())
        # Scale local values by sample counts for their aggregation.
        accuracy = self.computer.scale_metrics_by_sample_counts(accuracy)
        loss = self.computer.scale_metrics_by_sample_counts(loss)
        # Flatten shareable values, ordered and filled-out.
        share_values = [
            *[accuracy.get(group, 0.0) for group in self.groups],
            *[loss.get(group, 0.0) for group in self.groups],
        ]
        # Return both sets of values.
        return share_values, local_values

    async def finalize_fairness_round(
        self,
        netwk: NetworkClient,
        values: List[float],
        secagg: Optional[Encrypter],
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Await updated loss weights from the server.
        await self._update_fairbatch_sampling_probas(netwk)
        # Recover raw accuracy and loss values for groups with local samples.
        groups = list(self.computer.g_data)
        accuracy = dict(zip(groups, values[: len(groups)]))
        loss = dict(zip(groups, values[len(groups) :]))
        # Compute local fairness measures.
        fairness = self.fairness_function.compute_from_group_accuracy(accuracy)
        f_type = self.fairness_function.f_type
        # Package and return accuracy and fairness metrics.
        metrics = {
            f"accuracy_{key}": val for key, val in accuracy.items()
        }  # type: Dict[str, Union[float, np.ndarray]]
        metrics.update({f"loss_{key}": val for key, val in loss.items()})
        metrics.update(
            {f"{f_type}_{key}": val for key, val in fairness.items()}
        )
        return metrics
