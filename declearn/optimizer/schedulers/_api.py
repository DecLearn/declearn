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

"""API-defining abstract base class for time-based learning rate schedulers."""

import abc
from typing import Any, ClassVar, Dict, Self

from declearn.utils import (
    access_registered,
    create_types_registry,
    register_type,
)

__all__ = [
    "Scheduler",
]


@create_types_registry(name="Scheduler")
class Scheduler(metaclass=abc.ABCMeta):
    """Abstract base class for time-based learning rate schedulers.

    Subclasses are expected to implement a variety of time-based
    rules for updating a learning rate (or a weight decay rate)
    along the steps of a stochastic gradient descent training.

    Attributes
    ----------
    All `Scheduler` classes expose the following three attributes:

    - `base`:
        Value of the base learning rate, assigned at instantation.
        Meaning may vary across subclasses.
    - `steps`:
        Counter of passed training steps. Incremented by `get_next_rate`.
    - `rounds`:
        Counter of passes training rounds. Incremented by `on_round_start`.

    Abstract
    --------
    The following attribute and method require to be overridden
    by any non-abstract child class of `Scheduler`:

    - name: str class attribute
        Name identifier of the class (should be unique across existing
        Scheduler classes). Also used for automatic types-registration
        of the class (see `Inheritance` section below).
    - compute_value(step: int, round_: int) -> float:
        Compute the scheduled value at a given step and round index.
        This should not have side effects, and is called by the main
        `get_next_rate` method with current indices.

    Extendable
    ----------
    The following methods may (and often should) be overloaded by subclasses:

    - get_config() -> Dict[str, Any]:
        Return a JSON-serializable config dict to this instance.
        This should be overloaded to add algorithm-specific parameters.
    - from_config(Dict[str, Any]) -> Self:
        Instantiate from a config dict.
        This may be overloaded if some config parameters require some
        pre-processing before being input to the `__init__` method.

    Overridable
    -----------
    The following methods may be overridden to implement side effects that
    should occur in addition to incrementing `steps` or `rounds` counters.
    For most algorithms, they can remain as-is; in all cases, super calls
    should not be forgotten.

    - get_next_rate() -> float:
        Compute the rate at the current time, and increment `steps`.
    - on_round_start() -> None:
        Mark that a new training round starts, incrementing `rounds`.
    - get_state() -> Dict[str, Any]:
        Return a JSON-serializable dict of inner state variables.
        This contains `steps` and `rounds` by default.
    - set_state(Dict[str, Any]) -> None:
        Assign inner state variables.
        This expects `steps` and `rounds` by default.

    Inheritance
    -----------
    When a subclass inheriting from `Scheduler` is declared, it is
    automatically registered under the "Scheduler" group using its
    class-attribute `name`. This can be prevented by adding `register=False`
    to the inheritance specs (e.g. `class MyCls(Scheduler, register=False)`).
    See `declearn.utils.register_type` for details on types registration.
    """

    name: ClassVar[str]
    """Name identifier of the class, unique across Scheduler classes."""

    def __init__(
        self,
        base: float,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        """
        self.base = base
        self.steps = 0
        self.rounds = -1

    def __init_subclass__(
        cls,
        register: bool = True,
    ) -> None:
        """Automatically type-register subclasses."""
        if register:
            register_type(cls, name=cls.name, group="Scheduler")

    def get_next_rate(
        self,
    ) -> float:
        """Return the rate to apply at the next step.

        Calling this method increments this instance's `steps` counter,
        and may update any algorithm-specific states.

        Returns
        -------
        rate:
            Value of the next (learning or weight decay) rate.
        """
        value = self.compute_value(step=self.steps, round_=self.rounds)
        self.steps += 1
        return value

    @abc.abstractmethod
    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        """Compute the scheduled value at a given step and round index.

        This method may use any attributes from this instance, but
        should **neither make use of nor have a side effect on steps
        and rounds counters** attributes.

        Parameters
        ----------
        step:
            Index of the step at which to compute the value.
            This starts from 0 and increases across steps.
        round_:
            Index of the round at which to compute the value.
            This starts from 0 and increases across rounds.

        Returns
        -------
        value:
            Scheduled value to use at the indicated time indices.
        """

    def on_round_start(
        self,
    ) -> None:
        """Perform any required action at the start of a training round.

        By default, this increments the `rounds` rounds counter attribute.
        """
        self.rounds += 1

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable configuration dict to this instance.

        Returns
        -------
        config:
            JSON-serializable dict of parameters to this instance.
        """
        return {"base": self.base}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate a Scheduler from its configuration dict.

        Parameters
        ----------
        config:
            Configuration dict, as output by the `get_config` method.

        Returns
        -------
        scheduler:
            Instance of this class, parameterized based on `config`.
        """
        return cls(**config)

    @staticmethod
    def from_specs(
        name: str,
        config: Dict[str, Any],
    ) -> "Scheduler":
        """Instantiate a Scheduler from specifications.

        Parameters
        ----------
        name: str
            Name based on which the scheduler can be retrieved.
            Available as a class attribute.
        config: dict[str, any]
            Configuration dict of the scheduler, that is to be
            passed to its `from_config` class constructor.

        Returns
        -------
        scheduler:
            `Scheduler` instance that matches input specs.
        """
        cls = access_registered(name, group="Scheduler")
        assert issubclass(cls, Scheduler)  # tested by access_registered
        return cls.from_config(config)

    def get_state(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this module's state(s).

        The counterpart to this method is the `set_state` one.

        Returns
        -------
        state: Dict[str, Any]
            JSON-serializable dict storing this module's inner state
            variables.
        """
        return {"steps": self.steps, "rounds": self.rounds}

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        """Load a state dict into an instantiated module.

        The counterpart to this method is the `get_state` one.

        Parameters
        ----------
        state: dict[str, any]
            Dict storing values to assign to this module's inner
            state variables.

        Raises
        ------
        KeyError
            If an expected state variable is missing from `state`.
        """
        self.steps = int(state["steps"])
        self.rounds = int(state["rounds"])
