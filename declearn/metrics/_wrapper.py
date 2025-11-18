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

"""Wrapper for an ensemble of Metric objects."""

from typing import Any, Dict, List, Optional, Self, Sequence, Tuple, Union

import numpy as np

from declearn.metrics._api import Metric, MetricState

__all__ = [
    "MetricInputType",
    "MetricSet",
]


MetricInputType = Union[Metric, str, Tuple[str, Dict[str, Any]]]


class MetricSet:
    """Wrapper for an ensemble of Metric objects.

    This class is designed to wrap together a collection of `Metric`
    instances (see `declearn.metric.Metric`), and expose the key API
    methods in a grouped fashion, i.e. internalizing the boilerplate
    loops on the metrics to update them based on a batch of inputs,
    gather their states, compute their end results, reset them, etc.

    This class also enables specifying an ensemble of metrics through
    a modular specification system, where each metric may be provided
    either as an instance, a name identifier string, or a tuple with
    both the former identifier and a configuration dict (enabling the
    use of non-default hyper-parameters).
    """

    def __init__(
        self,
        metrics: Sequence[MetricInputType],
    ) -> None:
        """Instantiate the grouped ensemble of Metric instances.

        Parameters
        ----------
        metrics: list[Metric, str, tuple(str, dict[str, any])]
            List of metrics to bind together. The metrics may be provided
            either as a Metric instance, a name identifier string, or a
            tuple with both a name identifier and a configuration dict.

        Raises
        ------
        TypeError
            If one of the input `metrics` elements is of improper type.
        KeyError
            If a metric name identifier fails to be mapped to a Metric class.
        RuntimeError
            If multiple metrics are of the same final type.
        """
        # REVISE: store metrics into a Dict and adjust labels when needed
        self.metrics: List[Metric] = []
        for metric in metrics:
            # reassign in new variable to avoid modifying loop variable
            _metric = metric
            if isinstance(_metric, str):
                _metric = Metric.from_specs(_metric)
            if isinstance(_metric, (tuple, list)):
                if (
                    (len(_metric) == 2)
                    and isinstance(_metric[0], str)
                    and isinstance(_metric[1], dict)
                ):
                    _metric = Metric.from_specs(*_metric)
            if not isinstance(_metric, Metric):
                raise TypeError(
                    "'MetricSet' inputs must be Metric instances, string "
                    "identifiers or (string identifier, config dict) tuples."
                )
            self.metrics.append(_metric)
        if len(set(type(m) for m in self.metrics)) < len(self.metrics):
            raise RuntimeError(
                "'MetricSet' cannot wrap multiple metrics of the same type."
            )

    @classmethod
    def from_specs(
        cls,
        metrics: Union[List[MetricInputType], "MetricSet", None],
    ) -> Self:
        """Type-check and/or transform inputs into a MetricSet instance.

        This classmethod is merely implemented to avoid duplicate and
        boilerplate code from polluting FL orchestrating classes.

        Parameters
        ----------
        metrics: list[MetricInputType] or MetricSet or None
            Inputs set up a MetricSet instance, instance to type-check
            or None, resulting in an empty MetricSet being returned.

        Returns
        -------
        metricset: MetricSet
            MetricSet instance, type-checked or instantiated from inputs.

        Raises
        ------
        TypeError
            If `metrics` is of improper type.

        Other exceptions may be raised when calling this class's `__init__`.
        """
        if metrics is None:
            metrics = cls([])
        if isinstance(metrics, list):
            metrics = cls(metrics)
        if not isinstance(metrics, cls):
            raise TypeError(
                f"'metrics' should be a `{cls.__name__}`, a valid list of "
                "Metric instances and/or specs to wrap into one, or None."
            )
        return metrics

    def get_result(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Compute the metric(s), based on the current state variables.

        Returns
        -------
        results: dict[str, float or numpy.ndarray]
            Dict of named result metrics, that may either be
            unitary float scores or numpy arrays.
        """
        results = {}
        for metric in self.metrics:
            results.update(metric.get_result())
        return results

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
            True labels or values that were to be predicted.
        y_pred: numpy.ndarray
            Predictions (scores or values) that are to be evaluated.
        s_wght: numpy.ndarray or None, default=None
            Optional sample weights to take into account in scores.
        """
        for metric in self.metrics:
            metric.update(y_true, y_pred, s_wght)

    def reset(
        self,
    ) -> None:
        """Reset the metric to its initial state."""
        for metric in self.metrics:
            metric.reset()

    def get_states(
        self,
    ) -> Dict[str, MetricState]:
        """Return a copy of the current state variables.

        This method is designed to expose and share partial results
        that may be aggregated with those of other instances of the
        same metric before computing overall results.

        Returns
        -------
        states:
            Dict of metric states that may be aggregated with their
            counterparts and re-assigned for finalization using the
            `set_states` then `get_result` methods of this object.
        """
        return {metric.name: metric.get_states() for metric in self.metrics}

    def set_states(
        self,
        states: Dict[str, MetricState],
    ) -> None:
        """Replace internal states with a copy of incoming ones.

        Parameters
        ----------
        states:
            Replacement states, as a compatible `MetricState` instance.

        Raises
        ------
        TypeError
            If any metric states are of improper type.
        """
        for metric in self.metrics:
            if metric.name in states:
                metric.set_states(states[metric.name])

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable configuration dict for this MetricSet."""
        cfg = [(metric.name, metric.get_config()) for metric in self.metrics]
        return {"metrics": cfg}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate a MetricSet from its configuration dict."""
        return cls(**config)
