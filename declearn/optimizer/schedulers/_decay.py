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

"""Standard time-based rate decay schedulers."""

import math
from typing import Any, Dict


from declearn.optimizer.schedulers._api import Scheduler

__all__ = [
    "ExponentialDecay",
    "InverseScaling",
    "LinearDecay",
    "PolynomialDecay",
    "RoundDecay",
    "StepDecay",
]


class ExponentialDecay(Scheduler):
    """Exponential decay scheduler."""

    name = "exponential-decay"

    def __init__(
        self,
        base: float,
        decay: float,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        decay:
            Factor of the exponential decay.
        """
        super().__init__(base)
        self.decay = decay

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        return self.base * math.exp(-self.decay * step)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["decay"] = self.decay
        return config


class InverseScaling(Scheduler):
    """Inverse-scaling decay scheduler."""

    name = "inverse-scaling"

    def __init__(
        self,
        base: float,
        rate: float,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        rate:
            Factor of the inverse-scaling decay.
        """
        super().__init__(base)
        self.rate = rate

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        if not step:
            return self.base
        return self.base / (step**self.rate)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["rate"] = self.rate
        return config


class LinearDecay(Scheduler):
    """Linear decay scheduler."""

    name = "linear-decay"

    def __init__(
        self,
        base: float,
        decay: float,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        decay:
            Factor of the linear decay.
        """
        super().__init__(base)
        self.decay = decay

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        if not step:
            return self.base
        return self.base / (self.decay * step)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["decay"] = self.decay
        return config


class PolynomialDecay(Scheduler):
    """Polynomial decay over rounds scheduler."""

    name = "polynomial-decay"

    def __init__(
        self,
        base: float,
        power: int,
        n_rounds: int,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        power:
            Power of the polynomial decay function.
        n_rounds:
            Maximum number of training rounds, beyond which the rate is null.
        """
        super().__init__(base)
        self.power = power
        self.n_rounds = n_rounds

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        decay = 1 - min(round_ / self.n_rounds, 1)
        return self.base * (decay**self.power)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["power"] = self.power
        config["n_rounds"] = self.n_rounds
        return config


class RoundDecay(Scheduler):
    """Linear decay over rounds scheduler."""

    name = "round-decay"

    def __init__(
        self,
        base: float,
        decay: float,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        decay:
            Factor of the linear decay that is to happen at each round start.
        """
        super().__init__(base)
        self.decay = decay

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        return self.base * (self.decay**round_)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["decay"] = self.decay
        return config


class StepDecay(Scheduler):
    """Linear step decay scheduler."""

    name = "step-decay"

    def __init__(
        self,
        base: float,
        decay: float,
        step_size: int,
    ) -> None:
        """Instantiate the scheduler.

        Parameters
        ----------
        base:
            Base value for the scheduled rate.
        decay:
            Factor of the linear decay.
        step_size:
            Number of steps to let go between each decay increment.
        """
        super().__init__(base)
        self.decay = decay
        self.step_size = step_size

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        step = step // self.step_size
        return self.base * (self.decay**step)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["decay"] = self.decay
        config["step_size"] = self.step_size
        return config
