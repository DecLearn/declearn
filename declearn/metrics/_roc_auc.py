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

"""Iterative and federative ROC AUC evaluation metrics."""

import dataclasses
from typing import Any, Dict, Optional, Self, Tuple, Type, Union

import numpy as np
import sklearn  # type: ignore
import sklearn.metrics  # type: ignore

from declearn.metrics._api import Metric, MetricState

__all__ = [
    "BinaryRocAUC",
]


@dataclasses.dataclass
class AurocState(MetricState):
    """Dataclass for Binary AUROC metric states with fixed thresholds."""

    tpos: np.ndarray
    tneg: np.ndarray
    fpos: np.ndarray
    fneg: np.ndarray
    thresh: np.ndarray

    @staticmethod
    def aggregate_thresh(
        val_a: np.ndarray,
        val_b: np.ndarray,
    ) -> np.ndarray:
        """Raise if thresholds differ, otherwise return them."""
        if (len(val_a) != len(val_b)) or np.any(val_a != val_b):
            raise ValueError(
                "Cannot aggregate AUROC states with distinct thresholds. "
                "To do so, use `AurocStateUnbound` containers."
            )
        return val_a

    def prepare_for_secagg(
        self,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        secagg = self.to_dict()
        clrtxt = {"thresh": secagg.pop("thresh")}
        return secagg, clrtxt


@dataclasses.dataclass
class AurocStateUnbound(AurocState):
    """Dataclass for Binary AUROC metric states with adaptive thresholds."""

    tpos: np.ndarray
    tneg: np.ndarray
    fpos: np.ndarray
    fneg: np.ndarray
    thresh: np.ndarray

    def aggregate(
        self,
        other: Self,
    ) -> Self:
        """Aggregate two binary AUROC metric states."""
        if not isinstance(other, AurocState):
            raise TypeError(
                f"'{self.__class__.__name__}.aggregate' expected a similar "
                f"type instance as input, received '{type(other)}'."
            )
        # Case when both states have the same thresholds.
        if (len(self.thresh) == len(other.thresh)) and np.all(
            self.thresh == other.thresh
        ):
            return self.__class__(
                tpos=self.tpos + other.tpos,
                tneg=self.tneg + other.tneg,
                fpos=self.fpos + other.fpos,
                fneg=self.fneg + other.fneg,
                thresh=self.thresh,
            )
        # Case when thresholds need to be combined and values interpolated.
        thresh = np.union1d(self.thresh, other.thresh)
        s_keys = ("tpos", "tneg", "fpos", "fneg")
        states_a = self._interpolate_roc_states(
            thresh_r=thresh,
            thresh_p=self.thresh,
            states_p={key: getattr(self, key) for key in s_keys},
        )
        states_b = self._interpolate_roc_states(
            thresh_r=thresh,
            thresh_p=other.thresh,
            states_p={key: getattr(other, key) for key in s_keys},
        )
        states = {key: states_a[key] + states_b[key] for key in s_keys}
        return self.__class__(**states, thresh=thresh)

    @staticmethod
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
                for key in states_r:  # noqa: PLC0206
                    states_r[key][idr] = states_p[key][idp]
                idp = min(idp + 1, max_p)
            # Case when the threshold is below the subset's minimum.
            elif idp == 0:
                for key in states_r:  # noqa: PLC0206
                    states_r[key][idr] = states_p[key][idp]
            # Case when the threshold is above the subset's maximum.
            elif thresh_p[max_p] < thr:
                for key in states_r:  # noqa: PLC0206
                    states_r[key][idr] = states_p[key][max_p]
            # Case when the threshold-indexed values must be interpolated.
            else:
                t_inf = thresh_p[idp - 1]
                t_sup = thresh_p[idp]
                for key in states_r:  # noqa: PLC0206
                    v_inf = states_p[key][idp - 1]
                    v_sup = states_p[key][idp]
                    states_r[key][idr] = thr * (
                        (v_sup - v_inf) / (t_sup - t_inf)
                    )
        # Return the interpolated states.
        return states_r

    def prepare_for_secagg(
        self,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        raise NotImplementedError(
            f"'{self.__class__.__name__}' does not support Secure Aggregation."
            " To use Secure Aggregation over AUROC curves, please set the "
            "initiating 'BinaryRocAUC' instance's 'bound' parameter to a "
            "tuple of bounding values (with an associated 'scale'), and use "
            "the same across all peers."
        )


class BinaryRocAUC(Metric[AurocState]):
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

    name = "binary-roc"
    state_cls = AurocState

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
        - Using the default `bound=None` enables the thresholds at which
          the ROC curve points are compute to vary dynamically based on
          inputs, but also based on input states to the `agg_states`
          method, that may come from a metric with different parameters.
        - Setting up explicit boundaries prevents thresholds from being
          adjusted at update time, and a ValueError will be raise by the
          `agg_states` method if inputs are adjusted to a distinct set
          of thresholds.
        """
        self.scale = scale
        self.label = label
        self.bound = bound
        super().__init__()

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"scale": self.scale, "label": self.label, "bound": self.bound}

    @property
    def prec(self) -> int:
        """Numerical precision of threshold values."""
        return int(f"{self.scale:.1e}".rsplit("-", 1)[-1])

    def build_initial_states(
        self,
    ) -> AurocState:
        if self.bound is None:
            bounds = (0.0, 1.0)
            aggcls: Type[AurocState] = AurocStateUnbound
        else:
            bounds = self.bound
            aggcls = AurocState
        thresh = self._build_thresholds(*bounds)
        names = ("tpos", "tneg", "fpos", "fneg")
        states = {key: np.zeros_like(thresh) for key in names}
        return aggcls(**states, thresh=thresh)

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
        tpos = self._states.tpos[::-1]
        tneg = self._states.tneg[::-1]
        fpos = self._states.fpos[::-1]
        fneg = self._states.fneg[::-1]
        # Compute true- and false-positive rates and derive AUC.
        with np.errstate(invalid="ignore"):
            tpr = np.nan_to_num(tpos / (tpos + fneg))
            fpr = np.nan_to_num(fpos / (fpos + tneg))
        auc = sklearn.metrics.auc(fpr, tpr)
        return {
            "tpr": tpr,
            "fpr": fpr,
            "thresh": self._states.thresh[::-1],
            "roc_auc": auc,
        }

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        s_wght: Optional[np.ndarray] = None,
    ) -> None:
        # Set up the scaled set of thresholds at which to estimate states.
        thresh = self._states.thresh
        if self.bound is None:
            thresh = self._build_thresholds(
                min(y_pred.min(), thresh[0]),
                max(y_pred.max(), thresh[-1]),
            )
            aggcls: Type[AurocState] = AurocStateUnbound
        else:
            aggcls = AurocState
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
        states = aggcls(
            tpos=(s_wght * (tru & pos)).sum(axis=0),
            tneg=(s_wght * (tru & ~pos)).sum(axis=0),
            fpos=(s_wght * ~(tru | pos)).sum(axis=0),
            fneg=(s_wght * (~tru & pos)).sum(axis=0),
            thresh=thresh,
        )
        # Aggregate these scores into the retained states.
        self._states += states

    def set_states(
        self,
        states: AurocState,
    ) -> None:
        # Prevent bounded instances from assigning unmatching inputs.
        if self.bound:
            if isinstance(states, AurocStateUnbound):
                states = AurocState.from_dict(states.to_dict())
            if not (
                (len(self._states.thresh) == len(states.thresh))
                and np.all(self._states.thresh == states.thresh)
            ):
                raise TypeError(
                    f"Cannot assign '{self.__class__.__name__}' states with "
                    "unmatching thresholds to an instance with bounded ones."
                )
        # Prevent unbounded instances from switching to bouded states.
        elif self.bound is None and not isinstance(states, AurocStateUnbound):
            states = AurocStateUnbound.from_dict(states.to_dict())
        # Delegate assignment to parent call (that raises on wrong type).
        return super().set_states(states)
