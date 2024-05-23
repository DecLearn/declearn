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

"""Cosine annealing rate decay schedulers."""

import math
from typing import Any, Dict


from declearn.optimizer.schedulers._api import Scheduler

__all__ = [
    "CosineAnnealing",
    "CosineAnnealingRounds",
    "CosineAnnealingWarmRestarts",
    "CosineAnnealingWarmRestartsRounds",
]


class CosineAnnealing(Scheduler):
    """Cosine Annealing scheduler over steps.

    This scheduler implements a cosine annealing that results
    in the scheduled rate decreasing with each and every step.
    """

    name = "cosine-annealing"

    def __init__(
        self,
        base: float,
        max_lr: float,
        n_steps: int,
    ) -> None:
        """Instantiate the cosine annealing scheduler.

        Parameters
        ----------
        base:
            Minimum learning rate towards which to decrease.
        max_lr:
            Maximum learning rate, from which to start.
        n_steps:
            Number of steps during which to carry the cosine
            annealing. Beyond that, constantly use the `base`
            value.
        """
        super().__init__(base)
        self.max_lr = max_lr
        self.n_steps = n_steps

    def compute_value(
        self,
        step: int,
    ) -> float:
        if step > self.n_steps:
            return self.base
        cosine = 1 + math.cos(step / self.n_steps * math.pi)
        return self.base + 0.5 * (self.max_lr - self.base) * cosine

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["max_lr"] = self.max_lr
        config["n_steps"] = self.n_steps
        return config


class CosineAnnealingRounds(Scheduler):
    """Cosine Annealing scheduler over rounds.

    This scheduler implements a cosine annealing that results
    in the scheduled rate decreasing at the start of each round.
    """

    name = "cosine-annealing-rounds"

    def __init__(
        self,
        base: float,
        max_lr: float,
        n_rounds: int,
    ) -> None:
        """Instantiate the cosine annealing scheduler.

        Parameters
        ----------
        base:
            Minimum learning rate towards which to decrease.
        max_lr:
            Maximum learning rate, from which to start.
        n_rounds:
            Number of rounds during which to carry the cosine
            annealing. Beyond that, constantly use the `base`
            value.
        """
        super().__init__(base)
        self.max_lr = max_lr
        self.n_rounds = n_rounds
        self.rounds = 0
        self._wrapped = CosineAnnealing(
            base=self.base, max_lr=self.max_lr, n_steps=self.n_rounds
        )

    def compute_value(
        self,
        step: int,
    ) -> float:
        return self._wrapped.compute_value(step=round_)

    def on_round_start(
        self,
    ) -> None:
        self.rounds += 1

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["max_lr"] = self.max_lr
        config["n_rounds"] = self.n_rounds
        return config


class CosineAnnealingWarmRestarts(Scheduler):
    """Cosine Annealing with Warm Restarts scheduler over steps.

    This scheduler implements a cosine annealing with warm restarts,
    that results in the scheduled rate decreasing with each and every
    step over fixed-length periods, at the end of which the rate is
    reset to (a factor of) its initial value and a new annealing cycle
    begins. This is based on the SGDR paper [1].

    References
    ----------
    [1] Loshchilov & Hutter (2016).
        SGDR: Stochastic Gradient Descent with Warm Restarts.
        https://arxiv.org/abs/1608.03983v5
    """

    name = "cosine-annealing-warm-restarts"

    def __init__(
        self,
        base: float,
        max_lr: float,
        period: int,
        t_mult: float = 1.0,
    ) -> None:
        """Instantiate the cosine annealing with warm restarts scheduler.

        Parameters
        ----------
        base:
            Minimum learning rate towards which to decrease.
        max_lr:
            Maximum learning rate, from which to start.
        period:
            Number of steps during which to carry the cosine
            annealing between warm restarts.
        t_mult:
            Multiplier by which to scale `max_lr` every time
            a warm restart occurs.
        """
        super().__init__(base)
        self.max_lr = max_lr
        self.period = period
        self.t_mult = t_mult
        self._cosine_annealing = CosineAnnealing(
            base=self.base, max_lr=self.max_lr, n_steps=self.period
        )

    def compute_value(
        self,
        step: int,
    ) -> float:
        cycle, cstep = divmod(step, self.period)
        self._cosine_annealing.max_lr = self.max_lr * (self.t_mult**cycle)
        return self._cosine_annealing.compute_value(step=cstep)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["max_lr"] = self.max_lr
        config["period"] = self.period
        config["t_mult"] = self.t_mult
        return config


class CosineAnnealingWarmRestartsRounds(CosineAnnealingWarmRestarts):
    """Cosine Annealing with Warm Restarts scheduler over rounds.

    This scheduler implements a cosine annealing with warm restarts,
    that results in the scheduled rate decreasing at the start of each
    round over fixed-length periods, at the end of which the rate is
    reset to (a factor of) its initial value and a new annealing cycle
    begins. This is based on the SGDR paper [1].

    References
    ----------
    [1] Loshchilov & Hutter (2016).
        SGDR: Stochastic Gradient Descent with Warm Restarts.
        https://arxiv.org/abs/1608.03983v5
    """

    name = "cosine-annealing-warm-restarts-rounds"

    def __init__(
        self,
        base: float,
        max_lr: float,
        period: int,
        t_mult: float = 1.0,
    ) -> None:
        """Instantiate the cosine annealing with warm restarts scheduler.

        Parameters
        ----------
        base:
            Minimum learning rate towards which to decrease.
        max_lr:
            Maximum learning rate, from which to start.
        period:
            Number of rounds during which to carry the cosine
            annealing between warm restarts.
        t_mult:
            Multiplier by which to scale `max_lr` every time
            a warm restart occurs.
        """
        super().__init__(base, max_lr=max_lr, period=period, t_mult=t_mult)
        self._value = self.base

    def on_round_start(
        self,
    ) -> None:
        self.rounds += 1

    def compute_value(
        self,
        step: int,
    ) -> float:
        return super().compute_value(step=self.rounds)
