# coding: utf-8

"""Base API and common examples of plug-in gradients-alteration algorithms."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Union


from declearn2.model.api import Vector
from declearn2.utils import create_types_registry, register_type


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

    The `declearn.optimizer.Optimizer` class defines the main tools
    and routines for computing and applying gradients-based updates.
    `OptiModule` instances are designed to be "plugged in" such an
    `Optimizer` instance to add intermediary operations between the
    moment gradients are obtained and that when they are applied as
    updates. Note that learning-rate use and optional (decoupled)
    weight-decay mechanisms are implemented at `Optimizer` level.

    Abstract:
    --------
    The following attribute and method require to be overridden
    by any non-abstract child class of `OptiModule`:

    name: str class attribute
        Keyword naming this module. This has an effect when passing
        auxiliary variables (see section below) between synchronous
        client/server modules, which should share the *same* name.
    run(gradients: Vector) -> Vector:
        Apply an adaptation algorithm to input gradients and return
        them. This is the main method for any `OptiModule`.

    Overridable:
    -----------
    The following methods may be overridden to implement information-
    passing and parallel behaviors between client/server module pairs.
    As defined at `OptiModule` level, they have no effect and may thus
    be safely ignored when implementing self-contained algorithms.

    collect_aux_var() -> Optional[Dict[str, Any]]:
        Emit a JSON-serializable dict of auxiliary variables,
        to be received by a counterpart of this module on the
        other side of the client/server relationship.
    process_aux_var(Dict[str, Any]) -> None:
        Process a dict of auxiliary variables, received from
        a counterpart to this module on the other side of the
        client/server relationship.
    """

    name: str = NotImplemented

    @abstractmethod
    def run(
            self,
            gradients: Vector,
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


create_types_registry("OptiModule", OptiModule)


@register_type(name="Momentum", group="OptiModule")
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

    name = "momentum"

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
            gradients: Vector,
        ) -> Vector:
        """Apply Momentum acceleration to input (pseudo-)gradients."""
        self.state = (self.beta * self.state) + ((1 - self.beta) * gradients)
        return self.state
