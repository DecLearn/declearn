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

"""Base API for loss regularization optimizer plug-ins."""

from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.model.api import Vector
from declearn.utils import (
    access_registered,
    create_types_registry,
    register_type,
)

__all__ = [
    "Regularizer",
]


@create_types_registry
class Regularizer(metaclass=ABCMeta):
    """Abstract class defining an API to implement loss-regularizers.

    The `Regularizer` API is close to the `OptiModule` one, with the
    following differences:
    * Regularizers are meant to be applied prior to Modules, as a way
      to complete the computation of "raw" gradients.
    * Regularizers do not provide an API to share stateful variables
      between a server and its clients.

    The aim of this abstraction (which itself operates on the Vector
    abstraction, so as to provide framework-agnostic algorithms) is
    to enable implementing loss-regularization terms, rewritten as
    gradients-correction bricks, that can easily and modularly be
    plugged into optimization algorithms.

    The `declearn.optimizer.Optimizer` class defines the main tools
    and routines for computing and applying gradients-based updates.
    `Regularizer` instances are designed to be "plugged in" such an
    `Optimizer` instance, to be applied to the raw gradients prior
    to any further processing (e.g. adaptative scaling algorithms).

    Abstract
    --------
    name: str class attribute
        Name identifier of the class (should be unique across existing
        Regularizer classes). Also used for automatic types-registration
        of the class (see `Inheritance` section below).
    run(gradients: Vector, weights: Vector) -> Vector:
        Compute the regularization term's derivative from weights,
        and add it to the input gradients. This is the main method
        for any `Regularizer`.

    Overridable
    -----------
    on_round_start() -> None:
        Perform any required operation (e.g. resetting a state variable)
        at the start of a training round. By default, this method has no
        effect and mey thus be safely ignored when no behavior is needed.

    Inheritance
    -----------
    When a subclass inheriting from `Regularizer` is declared, it is
    automatically registered under the "Regularizer" group using its
    class-attribute `name`. This can be prevented by adding `register=False`
    to the inheritance specs (e.g. `class MyCls(Regularizer, register=False)`).
    See `declearn.utils.register_type` for details on types registration.
    """

    name: ClassVar[str] = NotImplemented

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automatically type-register Regularizer subclasses."""
        super().__init_subclass__(**kwargs)
        if register:
            register_type(cls, cls.name, group="Regularizer")

    def __init__(
        self,
        alpha: float = 0.01,
    ) -> None:
        """Instantiate the loss regularization term.

        Parameters
        ----------
        alpha: float, default=0.01
            Coefficient scaling the regularization term as part of the
            regularized loss function's formulation.
        """
        self.alpha = alpha

    @abstractmethod
    def run(
        self,
        gradients: Vector,
        weights: Vector,
    ) -> Vector:
        """Compute and add the regularization term's derivative to gradients.

        Parameters
        ----------
        gradients: Vector
            Input gradients to which the correction term is to be added.
        weights: Vector
            Model weights with respect to which gradients were computed,
            and based on which the regularization term should be derived.

        Returns
        -------
        gradients: Vector
            Modified input gradients. The output Vector should be
            fully compatible with the input one - only the values
            of the wrapped coefficients may have changed.
        """

    def on_round_start(
        self,
    ) -> None:
        """Perform any required action at the start of a training round."""
        return None

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return the regularizer's JSON-serializable dict configuration."""
        return {"alpha": self.alpha}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate a Regularizer from its configuration dict."""
        return cls(**config)

    @staticmethod
    def from_specs(
        name: str,
        config: Dict[str, Any],
    ) -> "Regularizer":
        """Instantiate a Regularizer from its specifications.

        Parameters
        ----------
        name: str
            Name based on which the regularizer can be retrieved.
            Available as a class attribute.
        config: dict[str, any]
            Configuration dict of the regularizer, that is to be
            passed to its `from_config` class constructor.
        """
        cls = access_registered(name, group="Regularizer")
        if not issubclass(cls, Regularizer):
            raise TypeError("Retrieved a non-Regularizer class.")
        return cls.from_config(config)
