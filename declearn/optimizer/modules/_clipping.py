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

"""Batch-averaged gradients clipping plug-in modules."""

from typing import Any, ClassVar, Dict

from declearn.model.api import Vector
from declearn.optimizer.modules._api import OptiModule

__all__ = ["L2Clipping"]


class L2Clipping(OptiModule):
    """Fixed-threshold L2-norm gradient clipping module.

    This module implements the following algorithm:
        Init(max_norm):
        Step(max_norm):
            norm = euclidean_norm(grads)
            clip = min(norm, max_norm)
            grads *= clip / max_norm

    In other words, (batch-averaged) gradients are clipped
    based on their L2 (euclidean) norm, based on a single,
    fixed threshold (as opposed to more complex algorithms
    that may use parameter-wise and/or adaptive clipping).

    This may notably be used to bound the contribution of
    batch-based gradients to model updates, notably so as
    to bound the sensitivity associated to that action.
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
        c_scale = (l2_norm / self.max_norm).minimum(1.0)
        return gradients * c_scale

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"max_norm": self.max_norm}
