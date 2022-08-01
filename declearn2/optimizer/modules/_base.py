# coding: utf-8

"""Base API and common examples of plug-in gradients-alteration algorithms."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Union


from declearn2.model.api import Vector


__all__ = [
    'MomentumModule',
    'OptiModule',
]


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

    def collect_aux_var(
            self,
        ) -> Optional[Dict[str, Any]]:
        """Return auxiliary variables that need to be shared between nodes."""
        return None

    def process_aux_var(
            self,
            aux_var: Dict[str, Any],
        ) -> None:
        """Update this module based on received shared auxiliary variables."""
        # API-defining method; pylint: disable=unused-argument
        return None


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
