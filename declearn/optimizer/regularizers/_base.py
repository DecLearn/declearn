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

"""Common plug-in loss-regularization plug-ins."""

from typing import ClassVar, Optional

from declearn.model.api import Vector
from declearn.optimizer.regularizers._api import Regularizer

__all__ = [
    "FedProxRegularizer",
    "LassoRegularizer",
    "RidgeRegularizer",
]


class FedProxRegularizer(Regularizer):
    """FedProx loss-regularization plug-in.

    The FedProx algorithm is implemented through this regularizer,
    that adds a proximal term to the loss function so as to handle
    heterogeneity across clients in a federated learning context.
    See paper [1].

    This regularizer implements the following term:

        loss += alpha / 2 * (weights - ref_wgt)^2
        w/ ref_wgt := weights at the 1st step of the round

    To do so, it applies the following correction to gradients:

        grads += alpha * (weights - ref_wgt)

    In other words, this regularizer penalizes weights' departure
    (as a result from local optimization steps) from their initial
    (shared) values.

    References
    ----------
    [1] Li et al., 2020.
        Federated Optimization in Heterogeneous Networks.
        https://arxiv.org/abs/1812.06127
    """

    name: ClassVar[str] = "fedprox"

    def __init__(
        self,
        alpha: float = 0.01,
    ) -> None:
        super().__init__(alpha)
        self.ref_wgt: Optional[Vector] = None

    def on_round_start(
        self,
    ) -> None:
        self.ref_wgt = None

    def run(
        self,
        gradients: Vector,
        weights: Vector,
    ) -> Vector:
        if self.ref_wgt is None:
            self.ref_wgt = weights
            return gradients
        correct = self.alpha * (weights - self.ref_wgt)
        return gradients + correct


class LassoRegularizer(Regularizer):
    """L1 (Lasso) loss-regularization plug-in.

    This regularizer implements the following term:

        loss += alpha * l1_norm(weights)

    To do so, it applies the following correction to gradients:

        grads += alpha * sign(weights)
    """

    name: ClassVar[str] = "lasso"

    def run(
        self,
        gradients: Vector,
        weights: Vector,
    ) -> Vector:
        correct = self.alpha * weights.sign()
        return gradients + correct


class RidgeRegularizer(Regularizer):
    """L2 (Ridge) loss-regularization plug-in.

    This regularizer implements the following term:

        loss += alpha * l2_norm(weights)

    To do so, it applies the following correction to gradients:

        grads += alpha * 2 * weights
    """

    name: ClassVar[str] = "ridge"

    def run(
        self,
        gradients: Vector,
        weights: Vector,
    ) -> Vector:
        correct = 2 * self.alpha * weights
        return gradients + correct
