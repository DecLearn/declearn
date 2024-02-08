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

import dataclasses
import warnings
from typing import Any, Dict, Optional, Tuple

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.aggregator._api import Aggregator, ModelUpdates
from declearn.aggregator._avg import AveragingAggregator
from declearn.model.api import Vector

__all__ = [
    "GradientMaskedAveraging",
]


@dataclasses.dataclass
class GMAModelUpdates(ModelUpdates):
    """Dataclass for GradientMaskedAveraging model updates."""

    up_sign: Optional[Vector] = None

    def aggregate(
        self,
        other: Self,
    ) -> Self:
        # Ensure updates' sign are defined and will be aggregated,
        # without having a side effect on 'self' nor on 'other'.
        if isinstance(other, GMAModelUpdates):
            if other.up_sign is None:
                other_dict = other.to_dict()
                other_dict["up_sign"] = other.updates.sign() * other.weights
                other = type(other)(**other_dict)
            if self.up_sign is None:
                self_dict = self.to_dict()
                self_dict["up_sign"] = self.updates.sign() * self.weights
                return other.aggregate(self.__class__(**self_dict))
        return super().aggregate(other)

    def prepare_for_secagg(
        self,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        data = self.to_dict()
        if self.up_sign is None:
            data["up_sign"] = self.updates.sign() * self.weights
        return data, None


class GradientMaskedAveraging(Aggregator[GMAModelUpdates]):
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

    name = "gradient-masked-averaging"
    updates_cls = GMAModelUpdates

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
        self._avg = AveragingAggregator(steps_weighted, client_weights)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["threshold"] = self.threshold
        return config

    def prepare_for_sharing(
        self,
        updates: Vector,
        n_steps: int,
    ) -> GMAModelUpdates:
        data = self._avg.prepare_for_sharing(updates, n_steps)
        return GMAModelUpdates(data.updates, data.weights)

    def finalize_updates(
        self,
        updates: GMAModelUpdates,
    ) -> Vector:
        # Average model updates.
        values = self._avg.finalize_updates(
            ModelUpdates(updates.updates, updates.weights)
        )
        # Return if signs were not computed, denoting a lack of aggregation.
        if updates.up_sign is None:
            return values
        # Compute the average direction, taken as an agreement score.
        scores = self._avg.finalize_updates(
            ModelUpdates(updates.up_sign, updates.weights)
        )
        scores = scores * scores.sign()
        # Derive masking scores, using the thresholding hyper-parameter.
        clip = (scores - self.threshold).sign().maximum(0.0)
        scores = (1 - clip) * scores + clip  # s = 1 if s > t else s
        # Correct outputs' magnitude and return them.
        return values * scores

    def compute_client_weights(  # pragma: no cover
        self,
        updates: Dict[str, Vector],
        n_steps: Dict[str, int],
    ) -> Dict[str, float]:
        """Compute weights to use when averaging a given set of updates.

        This method is DEPRECATED as of DecLearn v2.4.
        It will be removed in DecLearn 2.6 and/or 3.0.
        """
        # pylint: disable=duplicate-code
        warnings.warn(
            f"'{self.__class__.__name__}.compute_client_weights' was"
            " deprecated in DecLearn v2.4. It will be removed in DecLearn"
            " v2.6 and/or v3.0.",
            DeprecationWarning,
        )
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=DeprecationWarning)
            return self._avg.compute_client_weights(updates, n_steps)
