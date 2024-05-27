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

from typing import List, Optional


from declearn.communication.api import NetworkClient
from declearn.communication.utils import verify_server_message_validity
from declearn.fairness.api import (
    FairnessAccuracy,
    FairnessRoundQuery,
    FairnessRoundReply,
    FairnessControllerClient,
    FairnessSetupQuery,
    SecaggFairnessAccuracy,
)
from declearn.fairness.core import FairnessAccuracyComputer, FairnessDataset
from declearn.fairness.fairgrad._messages import (
    FairgradSetupQuery,
    FairgradWeights,
)
from declearn.messaging import Error, SerializedMessage
from declearn.secagg.api import Encrypter
from declearn.training import TrainingManager

__all__ = [
    "FairgradControllerClient",
]


class FairgradControllerClient(FairnessControllerClient):
    """Client-side controller to implement Fed-FairGrad."""

    setup_query_cls = FairgradSetupQuery

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._accuracy_computer = (
            None
        )  # type: Optional[FairnessAccuracyComputer]

    async def finalize_fairness_setup(
        self,
        netwk: NetworkClient,
        query: FairnessSetupQuery,
        manager: TrainingManager,
    ) -> TrainingManager:
        assert isinstance(manager.train_data, FairnessDataset)
        # Set up a controller to compute group-wise model accuracy.
        self._accuracy_computer = FairnessAccuracyComputer(manager.train_data)
        # Await initial loss weights from the server.
        await self._update_fairgrad_weights(netwk, manager)
        # Return the input TrainingManager.
        return manager

    async def fairness_round(
        self,
        netwk: NetworkClient,
        manager: TrainingManager,
        received: SerializedMessage[FairnessRoundQuery],
        secagg: Optional[Encrypter],
    ) -> None:
        query = await verify_server_message_validity(
            netwk, received, expected=FairnessRoundQuery
        )
        await self._compute_and_send_groupwise_accuracy(
            netwk, manager, query, secagg
        )
        await self._update_fairgrad_weights(netwk, manager)

    async def _compute_and_send_groupwise_accuracy(
        self,
        netwk: NetworkClient,
        manager: TrainingManager,
        query: FairnessRoundQuery,
        secagg: Optional[Encrypter],
    ) -> None:
        # Compute the count-weighted group-wise accuracy, handling exceptions.
        try:
            accuracy = self._compute_groupwise_accuracy(manager, query)
        except Exception as exc:  # pylint: disable=broad-except
            manager.logger.error(
                "Exception raised when computing group-wise accuracy: %s", exc
            )
            await netwk.send_message(Error(repr(exc)))
            raise RuntimeError("Group accuracy computation failed.") from exc
        # Send the computed metrics to the server, optionally encrypted.
        manager.logger.info("Sending group-wise accuracy to the server.")
        reply = FairnessAccuracy(accuracy)
        if secagg is None:
            await netwk.send_message(reply)
        else:
            await netwk.send_message(
                SecaggFairnessAccuracy.from_cleartext_message(reply, secagg)
            )

    def _compute_groupwise_accuracy(
        self,
        manager: TrainingManager,
        query: FairnessRoundQuery,
    ) -> List[float]:
        """Compute (counts-weighted) accuracy over sensitive groups."""
        assert self._accuracy_computer is not None
        # Compute group-wise accuracy scores.
        accuracy = self._accuracy_computer.compute_groupwise_accuracy(
            model=manager.model,
            batch_size=query.batch_size,
            n_batch=query.n_batch,
            thresh=query.thresh,
        )
        # Multiply these scores by sample counts.
        accuracy = {
            key: val * self._accuracy_computer.counts[key]
            for key, val in accuracy.items()
        }
        # Return shareable group-wise values, ordered and filled out.
        return [accuracy.get(group, 0.0) for group in self.groups]

    async def _update_fairgrad_weights(
        self,
        netwk: NetworkClient,
        manager: TrainingManager,
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
            assert isinstance(manager.train_data, FairnessDataset)
            manager.train_data.set_sensitive_group_weights(
                weights,
                adjust_by_counts=True,
            )
        except (AssertionError, KeyError, TypeError) as exc:
            manager.logger.error(
                "Exception encountered when setting FairGrad weights: %s", exc
            )
            await netwk.send_message(Error(repr(exc)))
            raise RuntimeError("FairGrad weights update failed.") from exc
        # If things went well, ping the server back to indicate so.
        manager.logger.info("Updated FairGrad weights.")
        await netwk.send_message(FairnessRoundReply())
