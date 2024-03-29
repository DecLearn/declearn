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

"""Iterative and federative classification evaluation metrics."""

import dataclasses
from typing import Any, Collection, Dict, Optional, Union

import numpy as np
import sklearn  # type: ignore
import sklearn.metrics  # type: ignore

from declearn.metrics._api import Metric, MetricState
from declearn.metrics._utils import safe_division

__all__ = [
    "BinaryAccuracyPrecisionRecall",
    "MulticlassAccuracyPrecisionRecall",
]


@dataclasses.dataclass
class BinaryConfmat(MetricState):
    """Binary confusion matrix 'MetricState'."""

    tpos: float = 0.0
    tneg: float = 0.0
    fpos: float = 0.0
    fneg: float = 0.0


class BinaryAccuracyPrecisionRecall(Metric[BinaryConfmat]):
    """Binary classification accuracy, precision and recall metrics.

    This metric applies to binary classifier, and computes the (opt.
    weighted) amount of true positives (TP), true negatives (TN),
    false positives (FP) and false negatives (FN) predictions over
    time, from which basic evaluation metrics may be derived.

    Computed metrics are the following:

    * accuracy: float
        Accuracy of the classifier, i.e. P(pred==true).
        Formula: (TP + TN) / (TP + TN + FP + FN)
    * precision: float
        Precision score, i.e. P(true=1|pred=1).
        Formula: TP / (TP + FP)
    * recall: float
        Recall score, i.e. P(pred=1|true=1).
        Formula: TP / (TP + FN)
    * f-score: float
        F1-score, i.e. harmonic mean of precision and recall.
        Formula: (2*TP) / (2*TP + FP + FN)
    * confusion: 2-d numpy.ndarray
        Confusion matrix of predictions. Values: [[TN, FP], [FN, TP]]
    """

    name = "binary-classif"
    state_cls = BinaryConfmat

    def __init__(
        self,
        thresh: float = 0.5,
        label: Union[int, str] = 1,
    ) -> None:
        """Instantiate the binary accuracy / precision / recall metrics.

        Parameters
        ----------
        thresh: float, default=.5
            Threshold value around which to binarize input predictions.
        label: int or str, default=1
            Value of the positive labels in input true-label arrays.
        """
        self.thresh = thresh
        self.label = label
        super().__init__()

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"thresh": self.thresh, "label": self.label}

    def build_initial_states(
        self,
    ) -> BinaryConfmat:
        return BinaryConfmat()

    def get_result(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Unpack state variables for code readability.
        tpos = self._states.tpos
        tneg = self._states.tneg
        fpos = self._states.fpos
        fneg = self._states.fneg
        # Compute metrics, catching division-by-zero errors (replace with 0.0).
        scores = {
            "accuracy": safe_division(tpos + tneg, tpos + tneg + fpos + fneg),
            "precision": safe_division(tpos, tpos + fpos),
            "recall": safe_division(tpos, tpos + fneg),
            "f-score": safe_division(tpos + tpos, tpos + tpos + fpos + fneg),
        }  # type: Dict[str, Union[float, np.ndarray]]
        # Add the confusion matrix and return.
        scores["confusion"] = np.array([[tneg, fpos], [fneg, tpos]])
        return scores

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        s_wght: Optional[np.ndarray] = None,
    ) -> None:
        pos = y_true.flatten() == self.label
        tru = (y_pred.flatten() >= self.thresh) == pos
        s_wght = np.ones_like(tru) if s_wght is None else s_wght.flatten()
        self._states.tpos += float(sum(s_wght * (tru & pos)))
        self._states.tneg += float(sum(s_wght * (tru & ~pos)))
        self._states.fpos += float(sum(s_wght * ~(tru | pos)))
        self._states.fneg += float(sum(s_wght * (~tru & pos)))


@dataclasses.dataclass
class ClassifConfmat(MetricState):
    """Multiclass confusion matrix 'MetricState'."""

    confmat: np.ndarray

    @staticmethod
    def aggregate_confmat(
        val_a: np.ndarray,
        val_b: np.ndarray,
    ) -> np.ndarray:
        """Aggregate two confusion matrix arrays."""
        if val_a.shape != val_b.shape:
            raise ValueError(
                "Cannot aggregate confusion matrices with distinct shapes."
            )
        return val_a + val_b


class MulticlassAccuracyPrecisionRecall(Metric[ClassifConfmat]):
    """Multiclass classification accuracy, precision and recall metrics.

    This metric assumes that the evaluated classifier emits a score for
    each and every predictable label, and that the predicted label is
    that with the highest score. Alternatively, pre-selected labels (or
    one-hot encodings) may be passed as predictions.

    Computed metrics are the following:

    * accuracy: float
        Overall accuracy of the classifier, i.e. P(pred==true).
    * precision: 1-d numpy.ndarray
        Label-wise precision score, i.e. P(true=k|pred=k).
    * recall: 1-d numpy.ndarray
        Label-wise recall score, i.e. P(pred=k|true=k).
    * f-score: 1-d numpy.ndarray
        Label-wise f1-score, i.e. harmonic mean of precision and recall.
    * confusion: 2-d numpy.ndarray
        Confusion matrix of predictions, where C[i, j] indicates the
        (opt. weighted) number of samples belonging to label i that
        were predicted to belong to label j.
    """

    name = "multi-classif"
    state_cls = ClassifConfmat

    def __init__(
        self,
        labels: Collection[Union[int, str]],
    ) -> None:
        """Instantiate the multiclass accuracy/precision/recall metrics.

        Parameters
        ----------
        labels: collection of {int, str}
            Ordered set of possible labels.
        """
        self.labels = np.array(list(labels))
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return {"labels": self.labels.tolist()}

    def build_initial_states(
        self,
    ) -> ClassifConfmat:
        matrix = np.zeros((len(self.labels), len(self.labels)))
        return ClassifConfmat(matrix)

    def get_result(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Compute the metrics, silencing division-by-zero errors.
        confmat = self._states.confmat
        diag = np.diag(confmat)  # label-wise true positives
        pred = confmat.sum(axis=0)  # label-wise number of predictions
        true = confmat.sum(axis=1)  # label-wise number of labels (support)
        with np.errstate(invalid="ignore"):
            scores = {
                "accuracy": diag.sum() / confmat.sum(),
                "precision": diag / pred,
                "recall": diag / true,
                "f-score": 2 * diag / (pred + true),
            }
        # Convert NaNs resulting from zero-division to zero.
        scores = {k: np.nan_to_num(v, copy=False) for k, v in scores.items()}
        # Add a copy of the confusion matrix and return.
        scores["confusion"] = confmat.copy()
        return scores

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        s_wght: Optional[np.ndarray] = None,
    ) -> None:
        """Update the metric's internal state based on a data batch.

        Parameters
        ----------
        y_true: numpy.ndarray
            True labels that were to be predicted, as a 1-d array.
        y_pred: numpy.ndarray
            Predictions, either as a 1-d array of labels, or 2-d array
            of scores with shape `(len(y_true), len(self.labels))`. In
            the latter case, the label with the highest score is used
            as prediction (one-vs-all style).
        s_wght: numpy.ndarray or None, default=None
            Optional sample weights to take into account in scores.
        """
        if y_pred.ndim == 2:
            y_pred = self.labels[y_pred.argmax(axis=1)]
        elif y_pred.ndim != 1:
            raise TypeError("Expected 1-d or 2-d y_pred array.")
        self._states.confmat += sklearn.metrics.confusion_matrix(
            y_true, y_pred, labels=self.labels, sample_weight=s_wght
        )
