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

"""API-defining abstract base class for time-based learning rate schedulers."""

import abc
from typing import Any, Dict, Tuple

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.utils import (
    access_registered,
    access_registration_info,
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
    """

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
        self.step = 0

    def __init_subclass__(
        cls,
        register: bool = True,
    ) -> None:
        """Automatically type-register subclasses."""
        if register:
            register_type(cls, name=cls.__name__, group="Scheduler")

    def get_next_rate(
        self,
    ) -> float:
        """Return the rate to apply at the next step.

        Calling this method increments this instance's `step` counter,
        and may update any algorithm-specific states.

        Returns
        -------
        rate:
            Value of the next (learning or weight decay) rate.
        """
        value = self.compute_value(self.step)
        self.step += 1
        return value

    @abc.abstractmethod
    def compute_value(
        self,
        step: int,
    ) -> float:
        """Compute the current value, notwithstanding possible warmup.

        Step counter increment is handled as part of `get_next_rate`.

        Parameters
        ----------
        step:
            Index of the step at which to compute the value.
            This starts from 0 and increases across steps.

        Returns
        -------
        value:
            Value that is to be returned as the next rate.
        """

    def on_round_start(
        self,
    ) -> None:
        """Perform any required action at the start of a training round."""

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

    def get_specs(
        self,
    ) -> Tuple[str, Dict[str, Any]]:
        """Return serializable specifications of this instance.

        Returns
        -------
        specs:
            `(name, config)` tuple specifying this instance.
        """
        r_name = access_registration_info(self.__class__, group="Scheduler")[0]
        config = self.get_config()
        return r_name, config

    @staticmethod
    def from_specs(
        specs: Tuple[str, Dict[str, Any]],
    ) -> "Scheduler":
        """Instantiate a Scheduler from specifications.

        Parameters
        ----------
        specs:
            `(name, config)` tuple specifying a Scheduler.

        Returns
        -------
        scheduler:
            `Scheduler` instance, the class and config of which match `specs`.
        """
        r_name, config = specs
        cls = access_registered(r_name, group="Scheduler")
        assert issubclass(cls, Scheduler)
        return cls.from_config(config)
