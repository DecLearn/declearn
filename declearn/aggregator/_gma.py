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

"""Gradient Masked Averaging aggregation class."""

from typing import Any, ClassVar, Dict, Optional

from declearn.aggregator._base import AveragingAggregator
from declearn.model.api import Vector

__all__ = [
    "GradientMaskedAveraging",
]


class GradientMaskedAveraging(AveragingAggregator):
    """Gradient Masked Averaging Aggregator subclass.

    This class implements the gradient masked averaging algorithm
    proposed and analyzed in [1] that modifies the base averaging
    algorithm from FedAvg (and its derivatives) by correcting the
    averaged updates' magnitude based on the share of clients that
    agree on the updates' direction (coordinate-wise).

    The formula is the following:
        threshold in range(0, 1)  # hyperparameter
        grads = [grads_client_0, ..., grads_client_N]
        agree = abs(sum(sign(grads))) / len(grads)
        score = 1 if agree >= threshold else agree
        return score * avg(grads)

    Client-based and/or number-of-training-steps-based weighting
    may also be used, that will be taken into account both when
    averaging input gradients and computing the coordinate-wise
    average direction that make up for the agreement scores.

    References
    ----------
    [1] Tenison et alii, 2022.
        Gradient Masked Averaging for Federated Learning.
        https://arxiv.org/abs/2201.11986
    """

    name: ClassVar[str] = "gradient-masked-averaging"

    def __init__(
        self,
        threshold: float = 1.0,
        steps_weighted: bool = True,
        client_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Instantiate a gradient masked averaging aggregator.

        Parameters
        ----------
        threshold: float
            Threshold above which to round the coordinate-wise agreement
            score to 1. Must be in [0, 1] (FedAvg being the 0 edge case).
        steps_weighted: bool, default=True
            Whether to weight updates based on the number of optimization
            steps taken by the clients (relative to one another).
        client_weights: dict[str, float] or None, default=None
            Optional dict of client-wise base weights to use.
            If None, homogeneous base weights are used.

        Notes
        -----
        * One may specify `client_weights` and use `steps_weighted=True`.
          In that case, the product of the client's base weight and their
          number of training steps taken will be used (and unit-normed).
        * One may use incomplete `client_weights`. In that case, unknown-
          clients' base weights will be set to 1.
        """
        self.threshold = threshold
        super().__init__(steps_weighted, client_weights)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["threshold"] = self.threshold
        return config

    def aggregate(
        self,
        updates: Dict[str, Vector],
        n_steps: Dict[str, int],
    ) -> Vector:
        # Perform gradients' averaging.
        output = super().aggregate(updates, n_steps)
        # Compute agreement scores as to gradients' direction.
        g_sign = {client: grads.sign() for client, grads in updates.items()}
        scores = super().aggregate(g_sign, n_steps)
        scores = scores * scores.sign()
        # Derive masking scores, using the thresholding hyper-parameter.
        clip = (scores - self.threshold).sign().maximum(0.0)
        scores = (1 - clip) * scores + clip  # s = 1 if s > t else s
        # Correct outputs' magnitude and return them.
        return scores * output
