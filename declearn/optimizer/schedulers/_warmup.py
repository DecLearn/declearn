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

"""Warmup scheduler (wrapper)."""

import abc
from typing import Any, Dict, Optional, Self, Union

from declearn.optimizer.schedulers._api import Scheduler

__all__ = [
    "Warmup",
    "WarmupRounds",
]


class WarmupScheduler(Scheduler, register=False, metaclass=abc.ABCMeta):
    """ABC factoring some shared code for Warmup schedulers."""

    def __init__(
        self,
        base: Union[float, Scheduler],
        warmup: int,
    ) -> None:
        if isinstance(base, Scheduler):
            self.base = base.base
            self.wrapped: Optional[Scheduler] = base
        else:
            self.base = float(base)
            self.wrapped = None
        super().__init__(self.base)
        self.warmup = warmup

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["warmup"] = self.warmup
        if self.wrapped is not None:
            config["base"] = (self.wrapped.name, self.wrapped.get_config())
        return config

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        if isinstance(config["base"], (tuple, list)):
            config = config.copy()
            config["base"] = Scheduler.from_specs(*config["base"])
        return super().from_config(config)


class Warmup(WarmupScheduler):
    """Scheduler (wrapper) setting up a linear warmup over steps.

    This class may either be used as a simple `Scheduler` that
    implements a linear warmup towards a constant rate, or as
    a wrapper around another `Scheduler` instance that delays
    calls to the wrapped rule until after the initial linear
    warmup phase has been completed.
    """

    name = "warmup"

    def __init__(
        self,
        base: Union[float, Scheduler],
        warmup: int,
    ) -> None:
        """Instantiate the linear warmup scheduler.

        Parameters
        ----------
        base:
            Either a fixed base value or a wrapped scheduler to use
            once the warmup period is over.
        warmup:
            Number of steps over which to carry the linear warmup.
        """
        super().__init__(base=base, warmup=warmup)
        self.full_warmup_rounds = -1

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        if step < self.warmup:
            return self.base * (step + 1) / self.warmup
        if self.wrapped is None:
            return self.base
        return self.wrapped.compute_value(
            step=step - self.warmup, round_=round_ - self.full_warmup_rounds
        )

    def on_round_start(
        self,
    ) -> None:
        super().on_round_start()
        # Keep track of the number of rounds fully devoted to warmup.
        if self.steps < self.warmup:
            self.full_warmup_rounds += 1

    def get_state(
        self,
    ) -> Dict[str, Any]:
        state = super().get_state()
        state["full_warmup_rounds"] = self.full_warmup_rounds
        return state

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        if "full_warmup_rounds" not in state:  # pragma: no cover
            raise KeyError(
                f"Missing state parameter for '{self.__class__}': "
                "'full_warmup_rounds'."
            )
        super().set_state(state)
        self.full_warmup_rounds = state["full_warmup_rounds"]


class WarmupRounds(WarmupScheduler):
    """Scheduler (wrapper) setting up a linear warmup over rounds.

    This class may either be used as a simple `Scheduler` that
    implements a linear warmup towards a constant rate, or as
    a wrapper around another `Scheduler` instance that delays
    calls to the wrapped rule until after the initial linear
    warmup phase has been completed.
    """

    name = "warmup-rounds"

    def __init__(
        self,
        base: Union[float, Scheduler],
        warmup: int,
    ) -> None:
        """Instantiate the linear warmup scheduler.

        Parameters
        ----------
        base:
            Either a fixed base value or a wrapped scheduler to use
            once the warmup period is over.
        warmup:
            Number of rounds over which to carry the linear warmup.
        """
        super().__init__(base=base, warmup=warmup)
        self.warmup_steps = 0

    def get_next_rate(
        self,
    ) -> float:
        value = super().get_next_rate()
        # Keep track of the number of steps passed during warmup rounds.
        if self.rounds < self.warmup:
            self.warmup_steps += 1
        return value

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        if round_ < self.warmup:
            return self.base * (round_ + 1) / self.warmup
        if self.wrapped is None:
            return self.base
        return self.wrapped.compute_value(
            step=step - self.warmup_steps, round_=round_ - self.warmup
        )

    def get_state(
        self,
    ) -> Dict[str, Any]:
        state = super().get_state()
        state["warmup_steps"] = self.warmup_steps
        return state

    def set_state(
        self,
        state: Dict[str, Any],
    ) -> None:
        if "warmup_steps" not in state:  # pragma: no cover
            raise KeyError(
                f"Missing state parameter for '{self.__class__}': "
                "'warmup_steps'."
            )
        super().set_state(state)
        self.warmup_steps = state["warmup_steps"]
