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

"""Warmup scheduler (wrapper)."""

from typing import Any, Dict, Optional, Union

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.optimizer.schedulers._api import Scheduler

__all__ = [
    "Warmup",
]


class Warmup(Scheduler):
    """Scheduler (wrapper) setting up a linear warmup.

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
        if isinstance(base, Scheduler):
            self.base = base.base
            self.wrapped = base  # type: Optional[Scheduler]
        else:
            self.base = float(base)
            self.wrapped = None
        super().__init__(self.base)
        self.warmup = warmup

    def compute_value(
        self,
        step: int,
    ) -> float:
        if step < self.warmup:
            return self.base * (step + 1) / self.warmup
        if self.wrapped is None:
            return self.base
        return self.wrapped.compute_value(step - self.warmup)

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
