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

"""Base API and common examples of plug-in gradients-alteration algorithms."""

from typing import Any, ClassVar, Dict, Union

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
    gradient part of it, following the [pytorch implementation](\
    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html).
    The nesterov variant's implementation is equivalently adapted.

    This formulation is equivalent to the canonical one for constant learning
    rare (eta), with both approaches outputting:
        $$ w_{t+1} = w_t - \\eta \\sum_{k=1}^t \\beta^{t-k} \\nabla_k $$

    It may however yield differences when $\\eta$ changes through training:

    - (can.) $$ w_{t+1} = w_t - \\sum_{k=1}^t \\eta_k \\beta^{t-k} \\nabla_k $$
    - (ours) $$ w_{t+1} = w_t - \\eta_t \\sum_{k=1}^t \\beta^{t-k} \\nabla_k $$

    References
    ----------
    [1] Sutskever et. al., 2013.
        On the importance of initialization and momentum in deep learning
        https://proceedings.mlr.press/v28/sutskever13.pdf
    """

    name: ClassVar[str] = "momentum"

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
        self.velocity: Union[Vector, float] = 0.0

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"beta": self.beta, "nesterov": self.nesterov}

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        self.velocity = (self.beta * self.velocity) + gradients
        if self.nesterov:
            return (self.beta * self.velocity) + gradients
        return self.velocity

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {"velocity": self.velocity}

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        if "velocity" not in state:
            raise KeyError("Missing required state variable 'velocity'.")
        self.velocity = state["velocity"]


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

    name: ClassVar[str] = "ewma"

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
        self.state: Union[Vector, float] = 0.0

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"beta": self.beta}

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        self.state = (self.beta * self.state) + ((1 - self.beta) * gradients)
        return self.state

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {"state": self.state}

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        if "state" not in state:
            raise KeyError("Missing required state variable 'state'.")
        self.state = state["state"]


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

    References
    ----------
    [1] Zaheer and Reddi et al., 2018.
        Adaptive Methods for Nonconvex Optimization.
    """

    name: ClassVar[str] = "yogi-momentum"

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        sign = (self.state - gradients).sign()
        self.state = self.state - (sign * (1 - self.beta) * gradients)
        return self.state
