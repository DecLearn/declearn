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

"""Cosine annealing rate decay schedulers."""

import math
from typing import Any, Dict

from declearn.optimizer.schedulers._api import Scheduler

__all__ = [
    "CosineAnnealing",
    "CosineAnnealingWarmRestarts",
]


class CosineAnnealing(Scheduler):
    """Cosine Annealing scheduler.

    This scheduler implements a cosine annealing that results
    in the scheduled rate decreasing at each training step or
    round until a given index, beyond which it is constant.

    It can be considered as a specific kind of decay rule that
    is parameterized to reach a given constant value after a
    given duration.
    """

    name = "cosine-annealing"

    def __init__(
        self,
        base: float,
        max_lr: float,
        duration: int,
        step_level: bool = True,
    ) -> None:
        """Instantiate the cosine annealing scheduler.

        Parameters
        ----------
        base:
            Minimum learning rate towards which to decrease.
        max_lr:
            Maximum learning rate, from which to start.
        duration:
            Number of steps or rounds during which to carry the cosine
            annealing. Beyond that, constantly use the `base` value.
        step_level:
            Whether to decay after each step rather than after each round.
        """
        super().__init__(base)
        self.max_lr = max_lr
        self.duration = duration
        self.step_level = step_level

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        unit = step if self.step_level else round_
        if unit >= self.duration:
            return self.base
        cosine = 1 + math.cos(unit / self.duration * math.pi)
        return self.base + 0.5 * (self.max_lr - self.base) * cosine

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["max_lr"] = self.max_lr
        config["duration"] = self.duration
        config["step_level"] = self.step_level
        return config


class CosineAnnealingWarmRestarts(Scheduler):
    """Cosine Annealing with Warm Restarts scheduler over steps.

    This scheduler implements a cosine annealing with warm restarts,
    that results in the scheduled rate decreasing with each and every
    step or round over fixed-length periods, at the end of which the
    rate is reset to (a factor of) its initial value and a new annealing
    cycle begins. This is based on the SGDR paper [1].

    References
    ----------
    [1] Loshchilov & Hutter (2016).
        SGDR: Stochastic Gradient Descent with Warm Restarts.
        https://arxiv.org/abs/1608.03983v5
    """

    name = "cosine-annealing-warm-restarts"

    # pylint: disable-next=too-many-positional-arguments
    def __init__(
        self,
        base: float,
        max_lr: float,
        period: int,
        t_mult: float = 1.0,
        step_level: bool = True,
    ) -> None:
        """Instantiate the cosine annealing with warm restarts scheduler.

        Parameters
        ----------
        base:
            Minimum learning rate towards which to decrease.
        max_lr:
            Maximum learning rate, from which to start.
        period:
            Number of steps or rounds during which to carry the cosine
            annealing between warm restarts.
        t_mult:
            Multiplier by which to scale `max_lr` every time
            a warm restart occurs.
        step_level:
            Whether to decay after each step rather than after each round.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        super().__init__(base)
        self.max_lr = max_lr
        self.period = period
        self.t_mult = t_mult
        self.step_level = step_level
        self._cosine_annealing = CosineAnnealing(
            base=self.base,
            max_lr=self.max_lr,
            duration=self.period,
            step_level=self.step_level,
        )

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        cycle, cstep = divmod(step, self.period)
        self._cosine_annealing.max_lr = self.max_lr * (self.t_mult**cycle)
        return self._cosine_annealing.compute_value(step=cstep, round_=round_)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["max_lr"] = self.max_lr
        config["period"] = self.period
        config["t_mult"] = self.t_mult
        config["step_level"] = self.step_level
        return config
