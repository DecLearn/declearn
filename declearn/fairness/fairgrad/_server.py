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

"""Server-side Fed-FairGrad controller."""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from declearn.aggregator import Aggregator, SumAggregator
from declearn.communication.api import NetworkServer
from declearn.communication.utils import verify_client_messages_validity
from declearn.fairness.api import (
    FairnessAccuracy,
    FairnessRoundQuery,
    FairnessRoundReply,
    FairnessControllerServer,
    FairnessSetupQuery,
    SecaggFairnessAccuracy,
)
from declearn.fairness.core import instantiate_fairness_function
from declearn.fairness.fairgrad._messages import (
    FairgradSetupQuery,
    FairgradWeights,
)
from declearn.secagg.api import Decrypter
from declearn.secagg.messaging import aggregate_secagg_messages


class FairgradWeightsController:
    """Fairness controller to implement Faigrad optimization constraints."""

    # attrs serve readability; pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        counts: Dict[Tuple[Any, ...], int],
        f_type: str = "accuracy_parity",
        eta: float = 1e-2,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """Instantiate the FairGrad controller.

        Parameters
        ----------
        counts:
            Group-wise counts for all target-label and sensitive-attributes
            combinations, with format `{(label, *attrs): count}`.
        f_type:
            Name of the type of fairness based on which to constraint the
            optimization problem. By default, "accuracy_parity".
        eta:
            Learning rate of the controller, impacting the update rule for
            fairness constraints and associate weights.
            As a rule of thumb, it should be between 1/5 and 1/10 of the
            model weights optimizer's learning rate.
        eps:
            Epsilon value introducing some small tolerance to unfairness.
        **kwargs:
            Optional keyword arguments to the constants-computing function.
            Supported arguments:
                - `target: int|list[int]` for "equality_of_opportunity".
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        # Store some input parameters.
        self.counts = np.array(list(counts.values()))
        self.eta = eta
        self.eps = eps
        self.total = sum(self.counts)  # n_samples
        # Compute the fairness constraint constants.
        self.function = instantiate_fairness_function(
            f_type=f_type, counts=counts, **kwargs
        )
        # Initialize the gradient-weighting constraint parameters.
        n_groups = len(counts)
        self.f_k = np.zeros(n_groups)
        self._upper = np.zeros(n_groups)  # lambda_k^t
        self._lower = np.zeros(n_groups)  # delta_k^t

    def update_weights_based_on_accuracy(
        self,
        accuracy: Dict[Tuple[Any, ...], float],
    ) -> None:
        """Update the held fairness constraint and loss weight parameters.

        Parameters
        ----------
        accuracy:
            Dict containing group-wise accuracy metrics, formatted
            as `{group_k: sum_i(n_ik * accuracy_ik)}`.
        """
        f_k = self.function.compute_from_federated_group_accuracy(accuracy)
        self.f_k = np.array(list(f_k.values()))
        self._upper = np.maximum(
            0, self._upper + self.eta * (self.f_k - self.eps)
        )
        self._lower = np.maximum(
            0, self._lower - self.eta * (self.f_k + self.eps)
        )

    def get_current_weights(
        self,
        norm_nk: bool = True,
    ) -> List[float]:
        """Return current loss weights for each sensitive group.

        Parameters
        ----------
        norm_nk:
            Whether to divide output weights by `n_k`.
            This is useful in Fed-FairGrad to turn the
            base weights into client-wise ones.

        Returns
        -------
        weights:
            List of group-wise loss weights.
            Group definitions may be accessed as `groups` attribute.
        """
        # Compute P_k := P(sample \in group_k).
        p_tk = self.counts / self.total
        # Compute group weights as P_k + Sum_k'(c_k'^k (lambda_k' - delta_k')).
        ld_k = self._upper - self._lower
        c_kk = self.function.constants[1]
        weights = p_tk + np.dot(ld_k, c_kk)
        # Optionally normalize weights by group-wise total sample counts.
        if norm_nk:
            weights /= self.counts
        # Output the ordered list of group-wise loss weights.
        return weights.tolist()

    def get_current_fairness(
        self,
    ) -> Dict[Tuple[Any, ...], float]:
        """Return the group-wise current fairness level."""
        return {
            key: float(val) for key, val in zip(self.function.groups, self.f_k)
        }


class FairgradControllerServer(FairnessControllerServer):
    """Server-side controller to implement Fed-FairGrad."""

    def __init__(
        self,
        f_type: str,
        f_args: Optional[Dict[str, Any]] = None,
        eta: float = 1e-2,
        eps: float = 1e-6,
    ) -> None:
        """Instantiate the server-side Fed-FairGrad controller.

        Parameters
        ----------
        f_type:
            Name of the fairness function to evaluate and optimize.
        f_args:
            Optional dict of keyword arguments to the fairness function.
        eta:
            Learning rate of the controller, impacting the update rule for
            fairness constraints and associate weights.
            As a rule of thumb, it should be between 1/5 and 1/10 of the
            model weights optimizer's learning rate.
        eps:
            Epsilon value introducing some small tolerance to unfairness.
            This may be set to 0.0 to try and enforce absolute fairness.
        """
        super().__init__(f_type=f_type, f_args=f_args)
        self.weights_controller = (
            None
        )  # type: Optional[FairgradWeightsController]
        self._eta = eta
        self._eps = eps

    def prepare_fairness_setup_query(
        self,
    ) -> FairnessSetupQuery:
        return FairgradSetupQuery()

    async def finalize_fairness_setup(
        self,
        netwk: NetworkServer,
        counts: List[int],
        aggregator: Aggregator,
    ) -> Aggregator:
        # Set up the FairgradWeightsController.
        self.weights_controller = FairgradWeightsController(
            counts=dict(zip(self.groups, counts)),
            f_type=self.f_type,
            eta=self._eta,
            eps=self._eps,
            **self.f_args,
        )
        # Send initial loss weights to the clients.
        await self._send_fairgrad_weights(netwk)
        # Force the use of a SumAggregator.
        if not isinstance(aggregator, SumAggregator):
            warnings.warn(
                "Overriding Aggregator choice to a 'SumAggregator', "
                "due to the use of Fed-FairGrad.",
                category=RuntimeWarning,
            )
            aggregator = SumAggregator()
        return aggregator

    async def _send_fairgrad_weights(
        self,
        netwk: NetworkServer,
    ) -> None:
        """Send FairGrad sensitive group loss weights to clients.

        Await for clients to ping back that things went fine on their side.
        """
        netwk.logger.info("Sending FairGrad weights to clients.")
        assert self.weights_controller is not None
        weights = self.weights_controller.get_current_weights(norm_nk=True)
        await netwk.broadcast_message(FairgradWeights(weights=weights))
        received = await netwk.wait_for_messages()
        await verify_client_messages_validity(
            netwk, received, expected=FairnessRoundReply
        )

    async def fairness_round(
        self,
        netwk: NetworkServer,
        secagg: Optional[Decrypter],
    ) -> None:
        assert self.weights_controller is not None
        # Send a query to clients and await group-wise accuracy metrics.
        await netwk.broadcast_message(
            FairnessRoundQuery()  # TODO: receive a config and use it
        )
        received = await netwk.wait_for_messages()
        # When SecAgg is not set, expect and aggregate cleartext values.
        if secagg is None:
            replies = await verify_client_messages_validity(
                netwk, received, expected=FairnessAccuracy
            )
            accuracy = self._aggregate_cleartext_accuracy(replies)
        # When SecAgg is set, expect and secure-aggregate encrypted values.
        else:
            sec_rep = await verify_client_messages_validity(
                netwk, received, expected=SecaggFairnessAccuracy
            )
            accuracy = aggregate_secagg_messages(sec_rep, secagg).values
        # Compute global fairness and update FairGrad loss weights.
        self.weights_controller.update_weights_based_on_accuracy(
            accuracy=dict(zip(self.groups, accuracy))
        )
        # Send back the updated weights to the clients.
        await self._send_fairgrad_weights(netwk)

    def _aggregate_cleartext_accuracy(
        self,
        messages: Dict[str, FairnessAccuracy],
    ) -> List[float]:
        """Sum group-wise accuracy metrics received from clients."""
        accuracy = np.zeros(len(self.groups), dtype="float64")
        for message in messages.values():
            accuracy += np.asarray(message.values, dtype="float64")
        return accuracy.tolist()
