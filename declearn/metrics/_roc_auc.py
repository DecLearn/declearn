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

"""Iterative and federative ROC AUC evaluation metrics."""

from typing import Any, ClassVar, Dict, Optional, Tuple, Union

import numpy as np
import sklearn  # type: ignore
import sklearn.metrics  # type: ignore

from declearn.metrics._api import Metric

__all__ = [
    "BinaryRocAUC",
]


class BinaryRocAUC(Metric):
    """ROC AUC metric for binary classification.

    This metric applies to a binary classifier, and computes the (opt.
    weighted) amount of true positives (TP), true negatives (TN), false
    positives (FP) and false negatives (FN) predictions over time around
    a variety of thresholds; from which TP rate, FP rate and finally ROC
    AUC metrics are eventually derived.

    Computed metrics are the following:
    * fpr: 1-d numpy.ndarray
        True-positive rate values for a variety of thresholds.
        Formula: TP / (TP + FN), i.e. P(pred=1|true=1)
    * tpr: 1-d numpy.ndarray
        False-positive rate values for a variety of thresholds.
        Formula: FP / (FP + TN), i.e. P(pred=1|true=0)
    * thresh: 1-d numpy.ndarray
        Array of decision thresholds indexing the FPR and TPR.
    * roc_auc: float
        ROC AUC, i.e. area under the receiver-operator curve, score.

    Note that this class supports aggregating states from another
    BinaryRocAUC instance with different hyper-parameters into it,
    unless its `bound` parameter is set - in which case thresholds
    are not authorized to be dynamically updated, either at samples
    processing or states-aggregating steps.
    """

    name: ClassVar[str] = "binary-roc"

    def __init__(
        self,
        scale: float = 0.1,
        label: Union[int, str] = 1,
        bound: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Instantiate the binary ROC AUC metric.

        Parameters
        ----------
        scale: float, default=.1
            Granularity of the set of threshold values around which
            to binarize input predictions for fpr/tpr estimation.
        label: int or str, default=1
            Value of the positive labels in input true-label arrays.
        bound: (float, float) tuple or None, default=None
            Optional lower and upper bounds for threshold values. If
            set, disable adjusting the scale based on input values.
            If None, start with (0, 1) and extend the scale on both
            ends when input values exceed them.

        Notes
        -----
        Using the default `bound=None` enables the thresholds at which
        the ROC curve points are compute to vary dynamically based on
        inputs, but also based on input states to the `agg_states`
        method, that may come from a metric with different parameters.
        Setting up explicit boundaries prevents thresholds from being
        adjusted at update time, and a ValueError will be raise by the
        `agg_states` method if inputs are adjusted to a distinct set
        of thresholds.
        """
        self.scale = scale
        self.label = label
        self.bound = bound
        super().__init__()

    def get_config(self) -> Dict[str, Any]:
        return {"scale": self.scale, "label": self.label, "bound": self.bound}

    @property
    def prec(self) -> int:
        """Numerical precision of threshold values."""
        return int(f"{self.scale:.1e}".rsplit("-", 1)[-1])

    def _build_states(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        bounds = (0, 1) if self.bound is None else self.bound
        thresh = self._build_thresholds(*bounds)
        names = ("tpos", "tneg", "fpos", "fneg")
        states = {key: np.zeros_like(thresh) for key in names}
        states["thr"] = thresh
        return states  # type: ignore

    def _build_thresholds(
        self,
        lower: float,
        upper: float,
    ) -> np.ndarray:
        """Return a 1-d array of increasing threshold values."""
        t_min = np.floor(lower / self.scale)
        t_max = np.ceil(upper / self.scale)
        return (np.arange(t_min, t_max + 1) * self.scale).round(self.prec)

    def get_result(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        # Unpack state variables for code readability.
        tpos = self._states["tpos"][::-1]  # type: ignore
        tneg = self._states["tneg"][::-1]  # type: ignore
        fpos = self._states["fpos"][::-1]  # type: ignore
        fneg = self._states["fneg"][::-1]  # type: ignore
        # Compute true- and false-positive rates and derive AUC.
        with np.errstate(invalid="ignore"):
            tpr = np.nan_to_num(tpos / (tpos + fneg), copy=False)
            fpr = np.nan_to_num(fpos / (fpos + tneg), copy=False)
        auc = sklearn.metrics.auc(fpr, tpr)
        return {
            "tpr": tpr,
            "fpr": fpr,
            "thr": self._states["thr"][::-1],  # type: ignore
            "roc_auc": auc,
        }

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        s_wght: Optional[np.ndarray] = None,
    ) -> None:
        # Set up the scaled set of thresholds at which to estimate states.
        thresh = self._states["thr"]  # type: np.ndarray  # type: ignore
        if self.bound is None:
            thresh = self._build_thresholds(
                min(y_pred.min(), thresh[0]),
                max(y_pred.max(), thresh[-1]),
            )
        # Adjust inputs' shape if needed.
        y_pred = y_pred.reshape((-1, 1))
        y_true = y_true.reshape((-1, 1))
        s_wght = (
            np.ones_like(y_pred) if s_wght is None else s_wght.reshape((-1, 1))
        )
        # Compute threshold-wise prediction truthness values.
        pos = y_true == self.label
        tru = (y_pred >= thresh) == pos
        # Aggregate the former into threshold-wise TP/TN/FP/FN scores.
        states = {
            "tpos": (s_wght * (tru & pos)).sum(axis=0),
            "tneg": (s_wght * (tru & ~pos)).sum(axis=0),
            "fpos": (s_wght * ~(tru | pos)).sum(axis=0),
            "fneg": (s_wght * (~tru & pos)).sum(axis=0),
        }
        # Aggregate these scores into the retained states.
        thresh, states = _combine_roc_states(
            thresh,
            states,
            self._states["thr"],  # type: ignore
            self._states,  # type: ignore
        )
        self._states = states  # type: ignore
        self._states["thr"] = thresh

    def agg_states(
        self,
        states: Dict[str, Union[float, np.ndarray]],
    ) -> None:
        # Run sanity check on input states.
        for name in ("tpos", "tneg", "fpos", "fneg", "thr"):
            if name not in states:
                raise KeyError(f"Missing required state variable: '{name}'.")
            if not isinstance(states[name], np.ndarray):
                raise TypeError(f"Input state '{name}' is of unproper type.")
            if states[name].ndim != 1:  # type: ignore
                raise ValueError(f"Input state array '{name}' should be 1-d.")
        # Gather thresholds. Raise if they differ and self is locked.
        thr_own = self._states["thr"]  # type: np.ndarray  # type: ignore
        thr_oth = states["thr"]  # type: np.ndarray  # type: ignore
        if self.bound:
            if (len(thr_own) != len(thr_oth)) or np.any(thr_own != thr_oth):
                msg = "Input thresholds differ from bounded self ones."
                raise ValueError(msg)
        # Combine input states with self ones.
        thresh, states = _combine_roc_states(  # type: ignore
            thr_own, self._states, thr_oth, states  # type: ignore
        )
        self._states = states
        self._states["thr"] = thresh


def _combine_roc_states(
    thresh_a: np.ndarray,
    states_a: Dict[str, np.ndarray],
    thresh_b: np.ndarray,
    states_b: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Combine ROC states values, re-indexing them to thresholds if needed.

    Parameters
    ----------
    thresh_a: 1d-array
        1-d array of unique and sorted thresholds indexing `states_a`.
    states_a: dict[str, 1d-array]
        Dict of named 1-d arrays of state values, aligned on `thresh_a`.
    thresh_b: 1d-array
        1-d array of unique and sorted thresholds indexing `states_b`.
    states_b: dict[str, 1d-array]
        Dict of named 1-d arrays of state values, aligned on `thresh_b`.

    Returns
    -------
    thresh: 1d-array
        1-d array of unique and sorted thresholds computed as the union
        of `thresh_a` and `thresh_b`.
    states: dict[str, 1d-array]
        Dict of named 1-d arrays of state values, aligned on `thresh`,
        that are the sum of (interpolated) `states_a` and `states_b`.
    """
    # Case when thresholds are the same: simply sum up values and return.
    if (len(thresh_a) == len(thresh_b)) and np.all(thresh_a == thresh_b):
        states = {key: states_a[key] + states_b[key] for key in states_a}
        return thresh_a, states
    # Case when thresholds need alignment.
    thresh = np.union1d(thresh_a, thresh_b)
    states_a = _interpolate_roc_states(thresh, thresh_a, states_a)
    states_b = _interpolate_roc_states(thresh, thresh_b, states_b)
    states = {key: states_a[key] + states_b[key] for key in states_a}
    return thresh, states


def _interpolate_roc_states(
    thresh_r: np.ndarray,
    thresh_p: np.ndarray,
    states_p: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Interpolate ROC states values to fit given thresholds.

    Parameters
    ----------
    thresh_r: 1d-array
        1-d array of unique and sorted reference thresholds.
    thresh_p: 1d-array
        1-d array of unique and sorted partial thresholds.
        `thresh_p` must be a subset of `thresh_r`.
    states_p: dict[str, 1d-array]
        Dict of named 1-d arrays of state values, aligned on
        `thresh_p` and monotonically increasing or decreasing.

    Returns
    -------
    states_r: dict[str, 1d-array]
        Dict of names 1-d arrays of interpolated state values,
        aligned on `thresh_r`.
    """
    keys = {"tpos", "tneg", "fpos", "fneg"}.intersection(states_p)
    states_r = {key: np.zeros_like(thresh_r) for key in keys}
    max_p = len(thresh_p) - 1
    idp = 0
    for idr, thr in enumerate(thresh_r):
        # Case when the threshold exists in the partial subset.
        if thresh_p[idp] == thr:
            for key in states_r:
                states_r[key][idr] = states_p[key][idp]
            idp = min(idp + 1, max_p)
        # Case when the threshold is below the subset's minimum.
        elif idp == 0:
            for key in states_r:
                states_r[key][idr] = states_p[key][idp]
        # Case when the threshold is above the subset's maximum.
        elif thresh_p[max_p] < thr:
            for key in states_r:
                states_r[key][idr] = states_p[key][max_p]
        # Case when the threshold-indexed values must and can be interpolated.
        else:
            t_inf = thresh_p[idp - 1]
            t_sup = thresh_p[idp]
            for key in states_r:
                v_inf = states_p[key][idp - 1]
                v_sup = states_p[key][idp]
                states_r[key][idr] = thr * ((v_sup - v_inf) / (t_sup - t_inf))
    # Return the interpolated states.
    return states_r
