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

"""Batch-averaged gradients clipping plug-in modules."""

from typing import Any, ClassVar, Dict, TypeVar

from declearn.model.api import Vector
from declearn.optimizer.modules._api import OptiModule

__all__ = ["L2Clipping"]


T = TypeVar("T")


class L2Clipping(OptiModule):
    """Fixed-threshold, per-parameter L2-norm gradient clipping module.

    This module implements the following algorithm:

        Init(max_norm):
            assign max_norm
        Step(grads):
            norm = euclidean_norm(grads)  # parameter-wise
            clip = max(max_norm / norm, 1.0)
            grads *= clip

    In other words, (batch-averaged) gradients are clipped based on
    their parameter-wise L2 (euclidean) norm, and on a single fixed
    threshold (as opposed to more complex algorithms that may use
    parameter-wise and/or adaptive clipping thresholds).

    This is equivalent to calling `tensorflow.clip_by_norm` on each
    and every data array in the input gradients `Vector`, with
    `max_norm` as norm clipping threshold. If you would rather clip
    gradients based on their global norm, use the `L2GlobalClipping`
    module (only available in declearn >=2.3).

    This may notably be used to bound the contribution of batch-based
    gradients to model updates, notably so as to bound the sensitivity
    associated to that action. It may also be used to prevent exploding
    gradients issues.
    """

    name: ClassVar[str] = "l2-clipping"

    def __init__(
        self,
        max_norm: float = 1.0,
    ) -> None:
        """Instantiate the L2-norm gradient-clipping module.

        Parameters
        ----------
        max_norm: float, default=1.0
            Clipping threshold of the L2 (euclidean) norm of
            input (batch-averaged) gradients.
        """
        self.max_norm = max_norm

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        l2_norm = (gradients**2).sum() ** 0.5
        scaling = (self.max_norm / l2_norm).minimum(1.0)
        return gradients * scaling

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"max_norm": self.max_norm}


class L2GlobalClipping(OptiModule):
    """Fixed-threshold, global-L2-norm gradient clipping module.

    This module implements the following algorithm:

        Init(max_norm):
            assign max_norm
        Step(grads):
            norm = euclidean_norm(flatten_and_stack(grads))
            clip = max(max_norm / norm, 1.0)
            grads *= clip

    In other words, (batch-averaged) gradients are clipped based on
    the L2 (euclidean) norm of their *concatenated* values, so that
    if that norm is above the selected `max_norm` threshold, all
    gradients are scaled by the same factor.

    This is equivalent to the `tensorflow.clip_by_global_norm` and
    `torch.utils.clip_grad_norm_` utils. If you would rather clip
    gradients on a per-parameter basis, use the `L2Clipping` module.

    This may be used to bound the sensitivity of gradient-based model
    updates, and/or to prevent exploding gradients issues.
    """

    name: ClassVar[str] = "l2-global-clipping"

    def __init__(
        self,
        max_norm: float = 1.0,
    ) -> None:
        """Instantiate the L2-norm gradient-clipping module.

        Parameters
        ----------
        max_norm: float, default=1.0
            Clipping threshold of the L2 (euclidean) norm of
            concatenated input gradients.
        """
        self.max_norm = max_norm

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        # Handle the edge case of an empty input Vector.
        if not gradients.coefs:
            return gradients
        # Compute the total l2 norm of gradients.
        sum_of_squares = (gradients**2).sum()
        total_sum_of_squares = sum(
            type(gradients)({"norm": value})
            for value in sum_of_squares.coefs.values()
        )
        l2_norm = total_sum_of_squares**0.5
        # Compute and apply the associate scaling.
        scaling = (self.max_norm / l2_norm).minimum(1.0).coefs["norm"]
        return gradients * scaling

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"max_norm": self.max_norm}
