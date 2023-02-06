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

"""Adaptive algorithms for optimizers, implemented as plug-in modules."""

from typing import Any, ClassVar, Dict, Optional, Union

from declearn.model.api import Vector
from declearn.optimizer.modules._api import OptiModule
from declearn.optimizer.modules._momentum import EWMAModule, YogiMomentumModule

__all__ = [
    "AdaGradModule",
    "AdamModule",
    "RMSPropModule",
    "YogiModule",
]


class AdaGradModule(OptiModule):
    """Adaptative Gradient Algorithm (AdaGrad) module.

    This module implements the following algorithm:
        Init(eps):
            state = 0
        Step(grads):
            state += (grads ** 2)
            grads /= (sqrt(state) + eps)

    In other words, gradients (i.e. indirectly the learning rate)
    are scaled down by the square-root of the sum of the past
    squared gradients. See reference [1].

    References:
    [1] Duchi et al., 2012.
        Adaptive Subgradient Methods for Online Learning
        and Stochastic Optimization.
        https://jmlr.org/papers/v12/duchi11a.html
    """

    name: ClassVar[str] = "adagrad"

    def __init__(
        self,
        eps: float = 1e-7,
    ) -> None:
        """Instantiate the Adagrad gradients-adaptation module.

        Parameters
        ----------
        eps: float, default=1e-7
            Numerical-stability improvement term, added
            to the (divisor) adapative scaling term.
        """
        self.eps = eps
        self.state = 0.0  # type: Union[Vector, float]

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"eps": self.eps}

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        self.state = self.state + (gradients**2)
        scaling = (self.state**0.5) + self.eps
        return gradients / scaling

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


class RMSPropModule(OptiModule):
    """Root Mean Square Propagation (RMSProp) module.

    This module implements the following algorithm:
        Init(beta, eps):
            state = 0
        Step(grads, step):
            state = beta*state + (1-beta)*(grads**2)
            grads /= (sqrt(state) + eps)

    In other words, gradients (i.e. indirectly the learning rate)
    are scaled down by the square-root of the momentum-corrected
    sum of the past squared gradients. See reference [1].

    References:
    [1] Tieleman and Hinton, 2012.
        Lecture 6.5-rmsprop: Divide the Gradient by a Running
        Average of its Recent Magnitude.
    """

    name: ClassVar[str] = "rmsprop"

    def __init__(
        self,
        beta: float = 0.9,
        eps: float = 1e-7,
    ) -> None:
        """Instantiate the RMSProp gradients-adaptation module.

        Parameters
        ----------
        beta: float
            Beta parameter for the momentum correction
            applied to the adaptive scaling term.
        eps: float, default=1e-7
            Numerical-stability improvement term, added
            to the (divisor) adapative scaling term.
        """
        self.ewma = EWMAModule(beta=beta)
        self.eps = eps

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"beta": self.ewma.beta, "eps": self.eps}

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        v_t = self.ewma.run(gradients**2)
        scaling = (v_t**0.5) + self.eps
        return gradients / scaling

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return self.ewma.get_state()

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        self.ewma.set_state(state)


class AdamModule(OptiModule):
    """Adaptive Moment Estimation (Adam) module.

    This module implements the following algorithm:
        Init(beta_1, beta_2, eps):
            state_m = 0
            state_v = 0
        Step(grads, step):
            state_m = beta_1*state_m + (1-beta_1)*grads
            state_v = beta_2*state_v + (1-beta_2)*(grads**2)
            m_hat = state_m / (1 - beta_1**step)
            v_hat = state_v / (1 - beta_2**step)
            grads = state_m / (sqrt(v_hat) + eps)

    In other words, gradients are first momentum-corrected, as
    is the accumulated sum of squared past gradients. Both are
    bias-corrected, then the former are scaled down based upon
    the latter AdaGrad-style (indirectly adapting the learning
    rate) and returned. This is the Adam [1] algorithm.

    Optionally, the AMSGrad [2] algorithm may be implemented,
    with a similar formula but using the element-wise maximum
    of present-and-past v_hat values as a scaling factor. This
    guarantees that the learning rate is shrinked across time,
    at least from the point of view of this module (a warm-up
    schedule might for example counteract this).

    References:
    [1] Kingma and Ba, 2014.
        Adam: A Method for Stochastic Optimization.
        https://arxiv.org/abs/1412.6980
    [2] Reddi et al., 2018.
        On the Convergence of Adam and Beyond.
        https://arxiv.org/abs/1904.09237
    """

    name: ClassVar[str] = "adam"

    def __init__(
        self,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        amsgrad: bool = False,
        eps: float = 1e-7,
    ) -> None:
        """Instantiate the Adam gradients-adaptation module.

        Parameters
        ----------
        beta_1: float
            Beta parameter for the momentum correction
            applied to the input gradients.
        beta_2: float
            Beta parameter for the momentum correction
            applied to the adaptive scaling term.
        amsgrad: bool, default=False
            Whether to implement the AMSGrad algorithm
            rather than the base Adam one.
        eps: float, default=1e-7
            Numerical-stability improvement term, added
            to the (divisor) adapative scaling term.
        """
        self.ewma_1 = EWMAModule(beta=beta_1)
        self.ewma_2 = EWMAModule(beta=beta_2)
        self.steps = 0
        self.eps = eps
        self.amsgrad = amsgrad
        self.vmax = None  # type: Optional[Vector]

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {
            "beta_1": self.ewma_1.beta,
            "beta_2": self.ewma_2.beta,
            "amsgrad": self.amsgrad,
            "eps": self.eps,
        }

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        # Compute momentum-corrected state variables.
        m_t = self.ewma_1.run(gradients)
        v_t = self.ewma_2.run(gradients**2)
        # Apply bias correction to the previous terms.
        m_h = m_t / (1 - (self.ewma_1.beta ** (self.steps + 1)))
        v_h = v_t / (1 - (self.ewma_2.beta ** (self.steps + 1)))
        # Optionally implement the AMSGrad algorithm.
        if self.amsgrad:
            if self.vmax is not None:
                v_h = v_h.maximum(self.vmax)
            self.vmax = v_h
        # Compute and return the adapted gradients.
        gradients = m_h / ((v_h**0.5) + self.eps)
        self.steps += 1
        return gradients

    def get_state(
        self,
    ) -> Dict[str, Any]:
        state = {
            "steps": self.steps,
            "vmax": self.vmax,
        }  # type: Dict[str, Any]
        state["momentum"] = self.ewma_1.get_state()
        state["velocity"] = self.ewma_2.get_state()
        return state

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        for key in ("momentum", "velocity", "steps", "vmax"):
            if key not in state:
                raise KeyError(f"Missing required state variable '{key}'.")
        self.ewma_1.set_state(state["momentum"])
        self.ewma_2.set_state(state["velocity"])
        self.steps = state["steps"]
        self.vmax = state["vmax"]


class YogiModule(AdamModule):
    """Yogi additive adaptive moment estimation module.

    This module implements the following algorithm:
        Init(beta_1, beta_2, eps):
            state_m = 0
            state_v = 0
        Step(grads, step):
            state_m = beta_1*state_m + (1-beta_1)*grads
            sign_uv = sign(state_v - grads**2)
            state_v = state_v + sign_uv*(1-beta_2)*(grads**2)
            m_hat = state_m / (1 - beta_1**step)
            v_hat = state_v / (1 - beta_2**step)
            grads = state_m / (sqrt(v_hat) + eps)

    In other words, Yogi [1] implements the Adam [2] algorithm,
    but modifies the update rule of the 'v' state variable that
    is used to scale the learning rate.

    Note that this implementation allows combining the Yogi
    modification of Adam with the AMSGrad [3] one.

    References:
    [1] Zaheer and Reddi et al., 2018.
        Adaptive Methods for Nonconvex Optimization.
    [2] Kingma and Ba, 2014.
        Adam: A Method for Stochastic Optimization.
        https://arxiv.org/abs/1412.6980
    [3] Reddi et al., 2018.
        On the Convergence of Adam and Beyond.
        https://arxiv.org/abs/1904.09237
    """

    name: ClassVar[str] = "yogi"

    def __init__(
        self,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        amsgrad: bool = False,
        eps: float = 1e-7,
    ) -> None:
        """Instantiate the Yogi gradients-adaptation module.

        Parameters
        ----------
        beta_1: float
            Beta parameter for the momentum correction
            applied to the input gradients.
        beta_2: float
            Beta parameter for the (Yogi-specific) momentum
            correction applied to the adaptive scaling term.
        amsgrad: bool, default=False
            Whether to implement the Yogi modification on top
            of the AMSGrad algorithm rather than the Adam one.
        eps: float, default=1e-7
            Numerical-stability improvement term, added
            to the (divisor) adapative scaling term.
        """
        super().__init__(beta_1, beta_2, amsgrad=amsgrad, eps=eps)
        self.ewma_2 = YogiMomentumModule(beta=beta_2)
