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

"""Iterative and federative regression-specific evaluation metrics."""

from typing import ClassVar, Dict, Optional, Union

import numpy as np

from declearn.metrics._api import Metric

__all__ = ["R2"]


class R2(Metric):
    """R2 regression metric.

    This metric applies to a regression model, and computes the (opt.
    weighted) R^2 score.

    Computed metric is the following:
    * r2: float
        R^2 score, or coefficient of determination, averaged across samples.
        It is defined as the proportion of total sample variance explained by our
        model :
        $$R^2(y, \\hat{y})=1-\\frac{\\sum_{i=1}^n w_i \\left(y_i-\\hat{y}_i\\right)^2}
        {\\sum_{i=1}^n w_i \\left(y_i-\\bar{y}\\right)^2}$$

    Note :
    * This metric expects 1d-arrays, or arrays than can be reduced to 1-d
    * If the true variance is zero, we by convention return a perfect score
    if the expected variance is also zero, else return a score of 0.0
    * The R^2 score is not well-defined with less than two samples.

    Implementation details:
    * Our implmentation uses a different formula than the one above, to account for
    possible weights as well the iterative nature of our calculatations - most notably,
    the quality our estimation of $\\bar{y}$ increases over traning, and we watn to use
    the best estimation possible to provide the final R^2 result.
    * We thus use estimate the König-Huygens formula to express the true variance as :
    $$\\sum_{i=1}^n w_i \\left(y_i-\\bar{y}\\right)^2 = \\sum_i w_i y_i^2-\\frac{\\left(\\sum_i w_i y_i\\right)^2}{\\sum_i w_i}$$
    """

    name: ClassVar[str] = "r2"

    def _build_states(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "explained_variance": 0.0,
            "weighted_sum_of_sq": 0.0,
            "weighted_sample_count": 0.0,
            "weighted_sum": 0.0,
        }

    def get_result(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        if self._states["weighted_sample_count"] == 0:
            true_var = 0.0
        else :
            true_var = (
            self._states["weighted_sum_of_sq"]
            - self._states["weighted_sum"] ** 2
            / self._states["weighted_sample_count"]
        )
        if true_var == 0:
            if self._states["explained_variance"] == 0:
                return {self.name: 1.0}
            return {self.name: 0.0}
        result = (
            1
            - self._states["explained_variance"]
            / true_var
        )
        return {self.name: float(result)}

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        s_wght: Optional[np.ndarray] = None,
    ) -> None:
        # Set weights
        if s_wght is not None:
            s_wght = s_wght.squeeze()
            if s_wght.shape != (len(y_pred),):
                raise ValueError(
                    "Improper shape for 's_wght': should be a 1-d array "
                    "of sample-wise scalar weights."
                )
        else:
            s_wght = np.ones_like(y_pred).squeeze()
        # Update explained var
        var_pred = (s_wght * self._sum_to_1d(y_true - y_pred) ** 2).sum()
        self._states["explained_variance"] += var_pred
        # Update states needed for true var
        self._states["weighted_sample_count"] += s_wght.sum()
        self._states["weighted_sum"] += (
            s_wght * self._sum_to_1d(y_true)
        ).sum()
        self._states["weighted_sum_of_sq"] += (
            s_wght * self._sum_to_1d(y_true) ** 2
        ).sum()

    @staticmethod
    def _sum_to_1d(val: np.ndarray) -> np.ndarray:
        "Utility method to reduce an array of any shape to a 1-d array"
        while val.ndim > 1:
            val = val.sum(axis=-1)
        return val
