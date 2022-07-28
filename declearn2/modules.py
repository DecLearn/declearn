# coding: utf-8

"""Gradients-alteration modules."""

from abc import ABCMeta, abstractmethod
from typing import Optional, Union

from declearn2.model.api import Vector


class OptiModule(metaclass=ABCMeta):
    """Abstract class defining an API to implement gradients adaptation tools.

    The aim of this abstraction (which itself operates on the Vector
    abstraction, so as to provide framework-agnostic algorithms) is
    to enable implementing unitary gradients-adaptation bricks that
    can easily and modularly be composed into complex algorithms.
    """

    @abstractmethod
    def run(
            self,
            gradient: Vector,
        ) -> Vector:
        """Apply an adaptation algorithm to input gradients."""
        return NotImplemented


class MomentumModule(OptiModule):
    """Momentum gradient-acceleration module.

    This module impements the following algorithm:
        Init(beta):
            state = 0
        Step(grads):
            state = beta*state + (1-beta)*grads
            grads = state

    In other words, gradients are corrected by an exponentially-
    decaying moving-average of past gradients.
    """

    def __init__(
            self,
            beta: float = 0.9,
        ) -> None:
        """Instantiate the Momentum gradients-adaptation module.

        beta: float, default=0.9
            Coefficient parameterizing the (exponentially-
            decaying) moving average of input gradients.
        """
        if not isinstance(beta, float):
            raise TypeError("'beta' should be of type float.")
        if not 0 <= beta < 1:
            raise ValueError("'beta' value should be in [0, 1[.")
        self.beta = beta
        self.state = 0.  # type: Union[Vector, float]

    def run(
            self,
            gradient: Vector,
        ) -> Vector:
        """Apply Momentum acceleration to input (pseudo-)gradients."""
        self.state = (self.beta * self.state) + ((1 - self.beta) * gradient)
        return self.state


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

    def __init__(
            self,
            eps: float = 1e-7,
        ) -> None:
        """Instantiate the Adagrad gradients-adaptation module.

        eps: float, default=1e-7
            Numerical-stability improvement term, added
            to the (divisor) adapative scaling term.
        """
        self.eps = eps
        self.state = 0.  # type: Union[Vector, float]

    def run(
            self,
            gradient: Vector,
        ) -> Vector:
        """Apply Adagrad adaptation to input (pseudo-)gradients."""
        self.state = self.state + (gradient ** 2)
        scaling = (self.state ** .5) + self.eps
        return gradient / scaling


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

    def __init__(
            self,
            beta: float = 0.9,
            eps: float = 1e-7,
        ) -> None:
        """Instantiate the RMSProp gradients-adaptation module.

        beta: float
            Beta parameter for the momentum correction
            applied to the adaptive scaling term.
        eps: float, default=1e-7
            Numerical-stability improvement term, added
            to the (divisor) adapative scaling term.
        """
        self.mom = MomentumModule(beta=beta)
        self.eps = eps

    def run(
            self,
            gradient: Vector,
        ) -> Vector:
        """Apply RMSProp adaptation to input (pseudo-)gradients."""
        v_t = self.mom.run(gradient ** 2)
        scaling = (v_t ** .5) + self.eps
        return gradient / scaling


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

    def __init__(
            self,
            beta_1: float = 0.9,
            beta_2: float = 0.99,
            amsgrad: bool = False,
            eps: float = 1e-7,
        ) -> None:
        """Instantiate the Adam gradients-adaptation module.

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
        self.mom_1 = MomentumModule(beta=beta_1)
        self.mom_2 = MomentumModule(beta=beta_2)
        self.steps = 0
        self.eps = eps
        self.amsgrad = amsgrad
        self._vmax = None  # type: Optional[Vector]

    def run(
            self,
            gradient: Vector,
        ) -> Vector:
        """Apply Adam adaptation to input (pseudo-)gradients."""
        # Compute momentum-corrected state variables.
        m_t = self.mom_1.run(gradient)
        v_t = self.mom_2.run(gradient ** 2)
        # Apply bias correction to the previous terms.
        m_h = m_t / (1 - (self.mom_1.beta ** (self.steps + 1)))
        v_h = v_t / (1 - (self.mom_2.beta ** (self.steps + 1)))
        # Optionally implement the AMSGrad algorithm.
        if self.amsgrad:
            if self._vmax is not None:
                v_h = v_h.maximum(self._vmax)
            self._vmax = v_h
        # Compute and return the adapted gradients.
        gradient = m_h / ((v_h ** .5) + self.eps)
        self.steps += 1
        return gradient


class YogiMomentumModule(MomentumModule):
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

    def run(
            self,
            gradient: Vector,
        ) -> Vector:
        """Apply Momentum acceleration to input (pseudo-)gradients."""
        sign = (self.state - gradient).sign()
        self.state = self.state + (sign * (1 - self.beta) * gradient)
        return self.state


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

    def __init__(
            self,
            beta_1: float = 0.9,
            beta_2: float = 0.99,
            amsgrad: bool = False,
            eps: float = 1e-7,
        ) -> None:
        """Instantiate the Yogi gradients-adaptation module.

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
        self.mom_2 = YogiMomentumModule(beta=beta_2)
