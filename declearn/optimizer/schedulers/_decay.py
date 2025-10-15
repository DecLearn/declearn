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

"""Standard time-based rate decay schedulers."""

import abc
from typing import Any, Dict

from declearn.optimizer.schedulers._api import Scheduler

__all__ = [
    "ExponentialDecay",
    "InverseScaling",
    "LinearDecay",
    "PiecewiseDecay",
    "PolynomialDecay",
]


class DecayScheduler(Scheduler, register=False, metaclass=abc.ABCMeta):
    """ABC factoring some shared code for Decay schedulers."""

    def __init__(
        self,
        base: float,
        rate: float,
        step_level: bool = True,
    ) -> None:
        super().__init__(base=base)
        self.rate = rate
        self.step_level = step_level

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["rate"] = self.rate
        config["step_level"] = self.step_level
        return config


class ExponentialDecay(DecayScheduler):
    """Exponential decay scheduler.

    This scheduler multiplies the base learning rate by a `rate`
    factor after each training step or training round.
    """

    name = "exponential-decay"

    def __init__(
        self,
        base: float,
        rate: float,
        step_level: bool = True,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        rate:
            Factor by which to multiply `base` after each unit.
        step_level:
            Whether to decay after each step rather than after each round.
        """
        super().__init__(base=base, rate=rate, step_level=step_level)

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        index = step if self.step_level else round_
        return self.base * self.rate**index


class InverseScaling(DecayScheduler):
    """Inverse-scaling decay scheduler.

    This scheduler scales the base learning rate by the step's or
    round's index power the `rate` parameter.
    """

    name = "inverse-scaling"

    def __init__(
        self,
        base: float,
        rate: float,
        step_level: bool = True,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        rate:
            Power at which to put the inverse-step-index factor
            to scale the `base` value at a given step.
        step_level:
            Whether to decay after each step rather than after each round.
        """
        super().__init__(base=base, rate=rate, step_level=step_level)

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        index = (step if self.step_level else round_) + 1
        return self.base / (index**self.rate)


class LinearDecay(DecayScheduler):
    """Linear decay scheduler over steps.

    This scheduler linearly decreases the base learning rate
    at each step or round by substracting `decay * base` from it.
    Once the learning rate becomes null, it remains constant.
    """

    name = "linear-decay"

    def __init__(
        self,
        base: float,
        rate: float,
        step_level: bool = True,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        rate:
            Value of the linear decay, _i.e._ share of `base` to substract
            from it after each step or round.
        step_level:
            Whether to decay after each step rather than after each round.
        """
        super().__init__(base=base, rate=rate, step_level=step_level)

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        index = step if self.step_level else round_
        scale = max(1 - self.rate * index, 0)
        return self.base * scale


class PolynomialDecay(Scheduler):
    """Polynomial decay scheduler.

    This scheduler scales the base learning rate by the
    share of remaining steps of rounds (given a `limit`)
    power a given `power` order.
    """

    name = "polynomial-decay"

    def __init__(
        self,
        base: float,
        power: int,
        limit: int,
        step_level: bool = True,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        power:
            Power of the polynomial decay function.
        limit:
            Maximum number of training steps or rounds,
            beyond which the rate is null.
        step_level:
            Whether to decay after each step rather than after each round.
            This also conditions the interpretation of `limit`.
        """
        super().__init__(base=base)
        self.power = power
        self.limit = limit
        self.step_level = step_level

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        index = step if self.step_level else round_
        rate = 1 - min(index / self.limit, 1)
        return self.base * (rate**self.power)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["power"] = self.power
        config["limit"] = self.limit
        config["step_level"] = self.step_level
        return config


class PiecewiseDecay(DecayScheduler):
    """Piecewise-constant exponential decay scheduler.

    This scheduler implements exponential decay over training
    steps or rounds, but increments the decay every `step_size`
    steps or rounds.
    """

    name = "piecewise-decay"

    def __init__(
        self,
        base: float,
        rate: float,
        step_size: int,
        step_level: bool = True,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        rate:
            Factor of the exponential decay
        step_size:
            Number of steps or rounds to let go between each decay increment.
        step_level:
            Whether to decay after each step rather than after each round.
            This also conditions the interpretation of `step_size`.
        """
        super().__init__(base=base, rate=rate, step_level=step_level)
        self.step_size = step_size

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        index = (step if self.step_level else round_) // self.step_size
        return self.base * (self.rate**index)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["step_size"] = self.step_size
        return config
