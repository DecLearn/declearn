# coding: utf-8

"""Common plug-in loss-regularization plug-ins."""

from typing import Optional

import numpy as np

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

    References:
    [1] Li et al., 2020.
        Federated Optimization in Heterogeneous Networks.
        https://arxiv.org/abs/1812.06127
    """

    name = "fedprox"

    def __init__(
        self,
        alpha: float = 0.01,
    ) -> None:
        super().__init__(alpha)
        self.ref_wgt = None  # type: Optional[Vector]

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
        grads += alpha
    """

    name = "lasso"

    def run(
        self,
        gradients: Vector,
        weights: Vector,
    ) -> Vector:
        return gradients + self.alpha


class RidgeRegularizer(Regularizer):
    """L2 (Ridge) loss-regularization plug-in.

    This regularizer implements the following term:
        loss += alpha * l2_norm(weights)

    To do so, it applies the following correction to gradients:
        grads += alpha * 2 * abs(weights)
    """

    name = "ridge"

    def run(
        self,
        gradients: Vector,
        weights: Vector,
    ) -> Vector:
        correct = 2 * self.alpha * weights.apply_func(np.abs)
        return gradients + correct
