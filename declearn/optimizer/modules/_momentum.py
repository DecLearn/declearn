# coding: utf-8

"""Base API and common examples of plug-in gradients-alteration algorithms."""

from typing import Any, Dict, Union

from declearn.model.api import Vector
from declearn.optimizer.modules._api import OptiModule

__all__ = [
    "EWMAModule",
    "MomentumModule",
    "YogiMomentumModule",
]


class MomentumModule(OptiModule):
    """Momentum gradient-acceleration module.

    This module impements the following algorithm:
        Init(beta):
            velocity = 0
        Step(grads):
            velocity = beta * velocity + grads
            if nesterov:
                grads = beta * velocity + grads
            else:
                grads = velocity

    Note that this contrasts with the canonical implementation of momentum
    by Sutskever et. al. [1]. The learning rate is applied to the whole output
    of the algorithm above, in the Optmizer class, rather than only to the
    gradient part of it, following the [pytorch implementation]\
    (https://pytorch.org/docs/stable/generated/torch.optim.SGD.html).
    The nesterov variant's implementation is equivalently adapted.

    This formaluation is equivalent to the canonical one for constant learning
    rare (eta), with both approaches outputting:
    $$ w_{t+1} = w_t - \\eta \\sum_{k=1}^t \\beta^{t-k} \nabla_k $$
    It may however yield differences when $\\eta$ changes through training:
    (can.) $$ w_{t+1} = w_t - \\sum_{k=1}^t \\eta_k \\beta^{t-k} \\nabla_k $$
    (ours) $$ w_{t+1} = w_t - \\eta_t \\sum_{k=1}^t \\beta^{t-k} \\nabla_k $$

    References
    ----------
    [1] Sutskever et. al., 2013.
        On the importance of initialization and momentum in deep learning
        https://proceedings.mlr.press/v28/sutskever13.pdf
    """

    name = "momentum"

    def __init__(
        self,
        beta: float = 0.9,
        nesterov: bool = False,
    ) -> None:
        """Instantiate the Momentum gradients-adaptation module.

        Parameters
        ----------
        beta: float, default=0.9
            Momentum coefficient parameterizing the weight of the velocity.
        nesterov : bool, default=False
            Whether to use Nesterov-accelerated momentum.
        """

        if not isinstance(beta, float):
            raise TypeError("'beta' should be of type float.")
        if not 0 <= beta < 1:
            raise ValueError("'beta' value should be in [0, 1[.")
        self.beta = beta
        self.nesterov = nesterov
        self.velocity = 0.0  # type: Union[Vector, float]

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this module's parameters."""
        return {"beta": self.beta, "nesterov": self.nesterov}

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        """Apply Momentum acceleration to input (pseudo-)gradients."""
        self.velocity = (self.beta * self.velocity) + gradients
        if self.nesterov:
            return (self.beta * self.velocity) + gradients
        return self.velocity


class EWMAModule(OptiModule):
    """Exponentially Weighted Moving Average module.

    This module impements the following algorithm:
        Init(beta):
            state = 0
        Step(grads):
            state = beta*state + (1-beta)*grads
            grads = state

    In other words, gradients are corrected by an exponentially-
    decaying moving-average of past gradients.
    """

    name = "ewma"

    def __init__(
        self,
        beta: float = 0.9,
    ) -> None:
        """Instantiate the EWMA gradients-adaptation module.

        Parameters
        ----------
        beta: float, default=0.9
            Coefficient parameterizing the (exponentially-
            decaying) moving average of input gradients.
        """
        if not isinstance(beta, float):
            raise TypeError("'beta' should be of type float.")
        if not 0 <= beta < 1:
            raise ValueError("'beta' value should be in [0, 1[.")
        self.beta = beta
        self.state = 0.0  # type: Union[Vector, float]

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this module's parameters."""
        return {"beta": self.beta}

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        """Apply exponentially-weighted moving-average to the inputs."""
        self.state = (self.beta * self.state) + ((1 - self.beta) * gradients)
        return self.state


class YogiMomentumModule(EWMAModule):
    """Yogi-specific momentum gradient-acceleration module.

    This module impements the following algorithm:
        Init(beta):
            state = 0
        Step(grads):
            state = state + sign(state-grads)*(1-beta)*grads
            grads = state

    In other words, gradients are corrected in a somewhat-simlar
    fashion as in the base momentum formula, but so that the
    magnitude of the state update is merely a function of inputs
    rather than of both the inputs and the previous state [1].

    Note that this module is actually meant to be used to compute
    a learning-rate adaptation term based on squared gradients.

    References:
    [1] Zaheer and Reddi et al., 2018.
        Adaptive Methods for Nonconvex Optimization.
    """

    name = "yogi-momentum"

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        """Apply Momentum acceleration to input (pseudo-)gradients."""
        sign = (self.state - gradients).sign()
        self.state = self.state - (sign * (1 - self.beta) * gradients)
        return self.state
