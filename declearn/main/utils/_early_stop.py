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

"""Simple implementation of metric-based early stopping."""

from typing import Optional


from declearn.utils import dataclass_from_init


__all__ = [
    "EarlyStopping",
    "EarlyStopConfig",
]


class EarlyStopping:
    """Class implementing a metric-based early-stopping decision rule."""

    def __init__(
        self,
        tolerance: float = 0.0,
        patience: int = 1,
        decrease: bool = True,
        relative: bool = False,
    ) -> None:
        """Instantiate the early stopping criterion.

        Parameters
        ----------
        tolerance: float, default=0.
            Improvement value wrt the best previous value below
            which the metric is deemed to be non-improving.
            If negative, define a tolerance to punctual regression
            of the metric. If positive, define an "intolerance" to
            low improvements (in absolute or relative value).
        patience: int, default=1
            Number of consecutive non-improving epochs that trigger
            early stopping.
        decrease: bool, default=True
            Whether the monitored metric is supposed to decrease
            rather than increase with training.
        relative: bool, default=False
            Whether the `tolerance` threshold should be compared
            to the `(last - best) / best` improvement ratio rather
            than to the absolute improvement value `last - best`.
        """
        self.tolerance = tolerance
        self.patience = patience
        self.decrease = decrease
        self.relative = relative
        self._best_metric = None  # type: Optional[float]
        self._n_iter_stuck = 0

    def reset(
        self,
    ) -> None:
        """Reset the early-stopping criterion to its initial state."""
        self._best_metric = None
        self._n_iter_stuck = 0

    @property
    def keep_training(self) -> bool:
        """Whether training should continue as per this criterion."""
        return self._n_iter_stuck < self.patience

    def update(
        self,
        metric: float,
    ) -> bool:
        """Update the early-stopping decision based on a new value.

        Parameters
        ----------
        metric: float
            Value of the monitored metric at the current epoch.

        Returns
        -------
        keep_training: bool
            Whether training should continue.
        """
        # Case when the input metric is the first to be received.
        if self._best_metric is None:
            self._best_metric = metric
            return True
        # Otherwise, compute the metric's improvement and act consequently.
        diff = (metric - self._best_metric) * (-1 if self.decrease else 1)
        if diff > 0:
            self._best_metric = metric
        if self.relative:
            diff /= self._best_metric
        if diff < self.tolerance:
            self._n_iter_stuck += 1
        else:
            self._n_iter_stuck = 0
        return self.keep_training


EarlyStopConfig = dataclass_from_init(EarlyStopping, name="EarlyStopConfig")
