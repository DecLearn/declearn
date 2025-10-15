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

"""Cycle-based rate decay schedulers."""

from typing import Any, Dict

from declearn.optimizer.schedulers._api import Scheduler

__all__ = [
    "CyclicExpRange",
    "CyclicTriangular",
]


class CyclicTriangular(Scheduler):
    """Cyclic Learning Rate (CLR) scheduling policy with triangular cycle.

    This learning rate scaling policy implements a revolving cycle
    of rate increase and decrease between a minimum and a maximum
    value. Cycles have a triangular form, meaning the rate linearly
    increases then decreases between defined bounds at each cycle.
    Optionnally, the maximum learning rate may be divided by 2 at
    the end of each cycle, thus shrinking values' range.

    This corresponds to options 'triangular' and 'triangular2' from
    the original CLR paper [1], that are also used in the PyTorch
    implementation (`torch.optim.lr_scheduler.CyclicLR`). The third
    CLR variant (deemed 'exp_range') is implemented as the distinct
    `CyclicExpRange` class for readability and efficiency purposes.

    References
    ----------
    [1] Smith (2015).
        Cyclical Learning Rates for Training Neural Networks.
        https://arxiv.org/abs/1506.01186v6
    """

    name = "cyclic-triangular"

    def __init__(
        self,
        base: float,
        max_lr: float,
        stepsize: int,
        decay: bool = False,
    ) -> None:
        """Instantiate the "triangular" Cyclic Learning Rate scheduler.

        Parameters
        ----------
        base:
            Minimum learning rate to cycle from.
        max_lr:
            Maximum learning rate to cycle to.
        stepsize:
            Number of steps per half-cycle (i.e. to go from one
            extremum learning rate to the other).
        decay:
            Whether to halve the maximum learning rate at the
            end of each cycle (as in the 'triangular2' mode from
            the CLR paper and the PyTorch implementation).
        """
        super().__init__(base)
        self.max_lr = max_lr
        self.stepsize = stepsize
        self.decay = bool(decay)

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        # Compute the cycle and step-within-cycle indices.
        cycle, cstep = divmod(step, 2 * self.stepsize)
        # Compute the triangular cycle adjustment.
        scale = 1 - abs((cstep / self.stepsize) - 1)
        # Optionally scale down the maximum rate as cycles pass.
        if self.decay:
            scale /= 2**cycle
        # Compute and return the final value.
        value = self.base + scale * (self.max_lr - self.base)
        return value

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["max_lr"] = self.max_lr
        config["stepsize"] = self.stepsize
        config["decay"] = self.decay
        return config


class CyclicExpRange(Scheduler):
    """Cyclic Learning Rate (CLR) scheduling policy with exponential decay.

    This learning rate scaling policy implements a revolving cycle
    of rate increase and decrease between a minimum and a maximum
    value. Cycles have a triangular form, meaning the rate linearly
    increases then decreases between defined bounds at each cycle.
    Additionally, the minimum and maximum rates are exponentially
    decayed at the end of each cycle, thus shrinking both values
    and their range.

    This corresponds to the 'exp_range' option from the original CLR
    paper [1]. The other two options ('triangular' and 'triangular2')
    are implemented as a distinct `CyclicTriangular` class, both for
    readability and efficiency purposes.

    References
    ----------
    [1] Smith (2015).
        Cyclical Learning Rates for Training Neural Networks.
        https://arxiv.org/abs/1506.01186v6
    """

    name = "cyclic-exp-range"

    def __init__(
        self,
        base: float,
        max_lr: float,
        stepsize: int,
        decay: float,
    ) -> None:
        """Instantiate the "exp-range" Cyclic Learning Rate scheduler.

        Parameters
        ----------
        base:
            Minimum learning rate to cycle from.
        max_lr:
            Maximum learning rate to cycle to.
        stepsize:
            Number of steps per half-cycle (i.e. to go from one
            extremum learning rate to the other).
        decay:
            Factor by which to decay (meaning, multiply) both
            the minimum and maximum learning rates at the end
            of each cycle.
        """
        super().__init__(base)
        self.max_lr = max_lr
        self.stepsize = stepsize
        self.decay = decay

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        # Compute the decayed boundary values.
        decay = self.decay**step
        max_lr = self.max_lr * decay
        min_lr = self.base * decay
        # Compute the triangular cycle adjustment.
        cstep = step % (2 * self.stepsize)
        scale = 1 - abs((cstep / self.stepsize) - 1)
        # Compute and return the final value.
        value = min_lr + scale * (max_lr - min_lr)
        return value

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["max_lr"] = self.max_lr
        config["stepsize"] = self.stepsize
        config["decay"] = self.decay
        return config
