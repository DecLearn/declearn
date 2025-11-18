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

"""Iterative and federative generic evaluation metrics."""

import abc
import dataclasses
from typing import Dict, Optional, Union

import numpy as np

from declearn.metrics._api import Metric, MetricState
from declearn.metrics._utils import squeeze_into_identical_shapes

__all__ = [
    "Accuracy",
    "MeanMetric",
    "MeanAbsoluteError",
    "MeanSquaredError",
]


@dataclasses.dataclass
class MeanState(MetricState):
    """Generic 'MetricState' for average-based scalar metrics."""

    num_sum: float = 0.0
    divisor: float = 0.0


class MeanMetric(Metric[MeanState], register=False, metaclass=abc.ABCMeta):
    """Generic mean-aggregation metric template.

    This abstract class implements a template for Metric classes
    that rely on computing a sample-wise score and its average
    across iterative inputs.

    Abstract
    --------
    To implement such an actual Metric, inherit `MeanMetric` and define:

    - name: str class attribute:
        Name identifier of the Metric (should be unique across existing
        Metric classes). Used for automated type-registration and name-
        based retrieval. Also used to label output results.
    - metric_func(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        Method that computes a score from the predictions and labels
        associated with a given batch, that is to be aggregated into
        an average metric across all input batches.
    """

    state_cls = MeanState

    def build_initial_states(
        self,
    ) -> MeanState:
        return MeanState()

    @abc.abstractmethod
    def metric_func(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Compute the sample-wise metric for a single batch.

        This method is called by the `update` one, which adds optional
        sample-weighting and updates the metric's states to eventually
        compute the average metric over a sequence of input batches.

        Parameters
        ----------
        y_true: numpy.ndarray
            True labels or values that were to be predicted.
        y_pred: numpy.ndarray
            Predictions (scores or values) that are to be evaluated.

        Returns
        -------
        scores: numpy.ndarray
            Sample-wise metric value.
        """

    def get_result(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        if self._states.divisor == 0:
            return {self.name: 0.0}
        result = self._states.num_sum / self._states.divisor
        return {self.name: result}

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        s_wght: Optional[np.ndarray] = None,
    ) -> None:
        scores = self.metric_func(y_true, y_pred)
        if s_wght is None:
            self._states.num_sum += float(scores.sum())
            self._states.divisor += len(y_pred)
        else:
            s_wght = self._prepare_sample_weights(s_wght, len(y_pred))
            self._states.num_sum += float((s_wght * scores).sum())
            self._states.divisor += float(np.sum(s_wght))


class MeanAbsoluteError(MeanMetric):
    """Mean Absolute Error (MAE) metric.

    This metric applies to a regression model, and computes the (opt.
    weighted) mean sample-wise absolute error. Note that for inputs
    with multiple channels, the sum of absolute channel-wise errors
    is computed for each sample, and averaged across samples.

    Computed metric is the following:

    * mae: float
        Mean absolute error, averaged across samples (possibly
        summed over channels for (>=2)-dimensional inputs).
    """

    name = "mae"

    def metric_func(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        # Sample-wise (sum of) absolute error function.
        y_true, y_pred = squeeze_into_identical_shapes(y_true, y_pred)
        errors = np.abs(y_true - y_pred)
        while errors.ndim > 1:
            errors = errors.sum(axis=-1)
        return errors


class MeanSquaredError(MeanMetric):
    """Mean Squared Error (MSE) metric.

    This metric applies to a regression model, and computes the (opt.
    weighted) mean sample-wise squared error. Note that for inputs
    with multiple channels, the sum of squared channel-wise errors
    is computed for each sample, and averaged across samples.

    Computed metric is the following:

    * mse: float
        Mean squared error, averaged across samples (possibly
        summed over channels for (>=2)-dimensional inputs).
    """

    name = "mse"

    def metric_func(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        # Sample-wise (sum of) squared error function.
        y_true, y_pred = squeeze_into_identical_shapes(y_true, y_pred)
        errors = np.square(y_true - y_pred)
        while errors.ndim > 1:
            errors = errors.sum(axis=-1)
        return errors


class Accuracy(MeanMetric, register=False):
    """Metric container to compute classification accuracy iteratively.

    This metric applies to a single-label classification model,
    and computes the (opt. weighted) mean sample-wise accuracy.
    It requires true labels to be formatted as sample-wise scalar
    values (as opposed to one-hot encoded values).

    Computed metric is the following:

    * accuracy: float
        Mean accuracy, averaged across samples.
    """

    name = "accuracy"

    def __init__(
        self,
        thresh: float,
    ) -> None:
        super().__init__()
        self.thresh = thresh

    def metric_func(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        y_pred = (
            y_pred > self.thresh
            if (y_pred.ndim == 1) or (y_pred.shape[1] == 1)
            else y_pred.max(axis=1)
        )
        y_true, y_pred = squeeze_into_identical_shapes(y_true, y_pred)
        return y_pred == y_true
