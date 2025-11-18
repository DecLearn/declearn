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

"""Server-side Fed-FairBatch/FedFB controller."""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from declearn.aggregator import Aggregator, SumAggregator
from declearn.communication.api import NetworkServer
from declearn.communication.utils import verify_client_messages_validity
from declearn.fairness.api import FairnessControllerServer
from declearn.fairness.fairbatch._fedfb import setup_fedfb_controller
from declearn.fairness.fairbatch._messages import (
    FairbatchOkay,
    FairbatchSamplingProbas,
)
from declearn.fairness.fairbatch._sampling import setup_fairbatch_controller
from declearn.secagg.api import Decrypter

__all__ = [
    "FairbatchControllerServer",
]


class FairbatchControllerServer(FairnessControllerServer):
    """Server-side controller to implement Fed-FairBatch or FedFB.

    FairBatch [1] is a group-fairness-enforcing algorithm that relies
    on a specific form of loss reweighting mediated via the batching
    of samples for SGD steps. Namely, in FairBatch, batches are drawn
    by concatenating group-wise sub-batches, the size of which is the
    byproduct of the desired total batch size and group-wise sampling
    probabilities, with the latter being updated throughout training
    based on the measured fairness of the current model.

    This controller implements an adaptation of FairBatch for federated
    learning, that is limited to the setting of the original paper, i.e.
    a binary classification task on data that have a single and binary
    sensitive attribute.

    The `fedfb` instantiation parameter controls whether formulas from
    the original paper should be used for computing and updating group
    sampling probabilities (the default), or be replaced with variants
    introduced in the FedFB algorithm from paper [2].

    References
    ----------
    - [1]
        Roh et al. (2020).
        FairBatch: Batch Selection for Model Fairness.
        https://arxiv.org/abs/2012.01696
    - [2]
        Zeng et al. (2022).
        Improving Fairness via Federated Learning.
        https://arxiv.org/abs/2110.15545
    """

    algorithm = "fairbatch"

    def __init__(
        self,
        f_type: str,
        f_args: Optional[Dict[str, Any]] = None,
        alpha: float = 0.005,
        fedfb: bool = False,
    ) -> None:
        """Instantiate the server-side Fed-FairGrad controller.

        Parameters
        ----------
        f_type:
            Name of the fairness function to evaluate and optimize.
        f_args:
            Optional dict of keyword arguments to the fairness function.
        alpha:
            Hyper-parameter controlling the update rule for internal
            states and thereof sampling probabilities.
        fedfb:
            Whether to use FedFB formulas rather than to stick
            to those from the original FairBatch paper.
        """
        super().__init__(f_type=f_type, f_args=f_args)
        # Choose whether to use FedFB or FairBatch update rules.
        self._setup_function = (
            setup_fedfb_controller if fedfb else setup_fairbatch_controller
        )
        # Set up a temporary controller that will be replaced at setup time.
        self.sampling_controller = self._setup_function(
            f_type=self.f_type,
            counts={(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
            target=self.f_args.get("target", 1),
            alpha=alpha,
        )

    @property
    def fedfb(self) -> bool:
        """Whether this controller implements FedFB rather than Fed-FairBatch.

        FedFB is a published adaptation of FairBatch to the federated
        setting, that introduces changes to some FairBatch formulas.

        Fed-FairBatch is a DecLearn-introduced variant of FedFB that
        restores the original FairBatch formulas.
        """
        return self._setup_function is setup_fedfb_controller

    async def finalize_fairness_setup(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
        counts: List[int],
        aggregator: Aggregator,
    ) -> Aggregator:
        # Set up the FairbatchWeightsController.
        self.sampling_controller = self._setup_function(
            f_type=self.f_type,
            counts=dict(zip(self.groups, counts, strict=False)),
            target=self.f_args.get("target", 1),
            alpha=self.sampling_controller.alpha,
        )
        # Send initial loss weights to the clients.
        await self._send_fairbatch_probas(netwk)
        # Force the use of a SumAggregator.
        if not isinstance(aggregator, SumAggregator):
            warnings.warn(
                "Overriding Aggregator choice to a 'SumAggregator', "
                "due to the use of Fed-FairBatch.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            aggregator = SumAggregator()
        return aggregator

    async def _send_fairbatch_probas(
        self,
        netwk: NetworkServer,
    ) -> None:
        """Send FairBatch sensitive group sampling probabilities to clients.

        Await for clients to ping back that things went fine on their side.
        """
        netwk.logger.info(
            "Sending FairBatch sampling probabilities to clients."
        )
        probas = self.sampling_controller.get_sampling_probas()
        p_list = [probas[group] for group in self.groups]
        await netwk.broadcast_message(FairbatchSamplingProbas(p_list))
        received = await netwk.wait_for_messages()
        await verify_client_messages_validity(
            netwk, received, expected=FairbatchOkay
        )

    async def finalize_fairness_round(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
        values: List[float],
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Unpack group-wise accuracy and loss values.
        accuracy = dict(
            zip(self.groups, values[: len(self.groups)], strict=False)
        )
        loss = dict(zip(self.groups, values[len(self.groups) :], strict=False))
        # Update sampling probabilities and send them to clients.
        self.sampling_controller.update_from_federated_losses(loss)
        await self._send_fairbatch_probas(netwk)
        # Package and return accuracy, loss and fairness metrics.
        metrics: Dict[str, Union[float, np.ndarray]] = {
            f"accuracy_{key}": val for key, val in accuracy.items()
        }
        metrics.update({f"loss_{key}": val for key, val in loss.items()})
        f_func = self.sampling_controller.f_func
        fairness = f_func.compute_from_federated_group_accuracy(accuracy)
        metrics.update(
            {f"{self.f_type}_{key}": val for key, val in fairness.items()}
        )
        return metrics
