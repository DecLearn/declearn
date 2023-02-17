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

"""Iterative and federative R-Squared evaluation metric."""

from typing import ClassVar, Dict, Optional, Union

import numpy as np

from declearn.metrics._api import Metric

__all__ = [
    "RSquared",
]


class RSquared(Metric):
    """R^2 (R-Squared, coefficient of determination) regression metric.

    This metric applies to a regression model, and computes the (opt.
    weighted) R^2 score, also known as coefficient of determination.

    Computed metric is the following:
    * r2: float
        R^2 score, or coefficient of determination, averaged across samples.
        It is defined as the proportion of total sample variance explained
        by the regression model:
        * SSr = Sum((true - pred)^2)  # Residual sum of squares
        * SSt = Sum((true - mean(true))^2)  # Total sum of squares
        * R^2 = 1 - (SSr / SSt)

    Notes:
    - This metric expects 1d-arrays, or arrays than can be reduced to 1-d
    - If the true variance is zero, we by convention return a perfect score
      if the expected variance is also zero, else return a score of 0.0.
    - The R^2 score is not well-defined with less than two samples.

    Implementation details:
    - Since this metric is to be computed iteratively and with a single pass
      over a batched dataset, we use the König-Huygens formula to decompose
      the total sum of squares into a sum of terms that can be updated with
      summation for each batch received (as opposed to using an estimate of
      the mean of true values that would vary with each batch). This gives:
        wSST = Sum(weight * (true - mean(true))^2)  # initial definition
        wSST = Sum(weight * true^2) - (Sum(weight * true))^2 / Sum(weight)

    LaTeX formulas (with weights):
    - Canonical formula:
        $$R^2(y, \\hat{y})= 1 - \\frac{
            \\sum_{i=1}^n w_i \\left(y_i-\\hat{y}_i\\right)^2
        }{
            \\sum_{i=1}^n w_i \\left(y_i-\\bar{y}\\right)^2
        }$$
    - Decomposed weighted total sum of squares:
        $$\\sum_{i=1}^n w_i \\left(y_i-\\bar{y}\\right)^2 =
            \\sum_i w_i y_i^2
            - \\frac{\\left(\\sum_i w_i y_i\\right)^2}{\\sum_i w_i}
        $$
    """

    name: ClassVar[str] = "r2"

    def _build_states(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "sum_of_squared_errors": 0.0,
            "sum_of_squared_labels": 0.0,
            "sum_of_labels": 0.0,
            "sum_of_weights": 0.0,
        }

    def get_result(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Case when no samples were seen: return 0. by convention.
        if self._states["sum_of_weights"] == 0:
            return {self.name: 0.0}
        # Compute the (weighted) total sum of squares.
        ss_tot = (  # wSSt = sum(w * y^2) - (sum(w * y))^2 / sum(w)
            self._states["sum_of_squared_labels"]
            - self._states["sum_of_labels"] ** 2
            / self._states["sum_of_weights"]
        )
        ss_res = self._states["sum_of_squared_errors"]
        # Handle the edge case where SSt is null.
        if ss_tot == 0:
            return {self.name: 1.0 if ss_res == 0 else 0.0}
        # Otherwise, compute and return the R-squared metric.
        result = 1 - ss_res / ss_tot
        return {self.name: float(result)}

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        s_wght: Optional[np.ndarray] = None,
    ) -> None:
        # Verify sample weights' shape, or set up 1-valued ones.
        s_wght = self._prepare_sample_weights(s_wght, n_samples=len(y_pred))
        # Update the residual sum of squares. wSSr = sum(w * (y - p)^2)
        ss_res = (s_wght * self._sum_to_1d(y_true - y_pred) ** 2).sum()
        self._states["sum_of_squared_errors"] += ss_res
        # Update states that compose the total sum of squares.
        # wSSt = sum(w * y^2) - (sum(w * y))^2 / sum(w)
        y_true = self._sum_to_1d(y_true)
        self._states["sum_of_squared_labels"] += (s_wght * y_true**2).sum()
        self._states["sum_of_labels"] += (s_wght * y_true).sum()
        self._states["sum_of_weights"] += s_wght.sum()

    @staticmethod
    def _sum_to_1d(val: np.ndarray) -> np.ndarray:
        "Utility method to reduce an array of any shape to a 1-d array"
        while val.ndim > 1:
            val = val.sum(axis=-1)
        return val
