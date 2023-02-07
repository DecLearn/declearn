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

"""Wrapper for an ensemble of Metric objects."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.metrics._api import Metric

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
        metrics: List[MetricInputType],
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
        TypeError:
            If one of the input `metrics` elements is of improper type.
        KeyError:
            If a metric name identifier fails to be mapped to a Metric class.
        RuntimeError:
            If multiple metrics are of the same final type.
        """
        # REVISE: store metrics into a Dict and adjust labels when needed
        self.metrics = []  # type: List[Metric]
        for metric in metrics:
            if isinstance(metric, str):
                metric = Metric.from_specs(metric)
            if isinstance(metric, (tuple, list)):
                if (
                    (len(metric) == 2)
                    and isinstance(metric[0], str)
                    and isinstance(metric[1], dict)
                ):
                    metric = Metric.from_specs(*metric)
            if not isinstance(metric, Metric):
                raise TypeError(
                    "'MetricSet' inputs must be Metric instances, string "
                    "identifiers or (string identifier, config dict) tuples."
                )
            self.metrics.append(metric)
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
        TypeError:
            If `metrics` is of unproper type.
        Other exceptions may be raised when calling this class's `__init__`.
        """
        if metrics is None:
            metrics = cls([])
        if isinstance(metrics, list):
            metrics = cls(metrics)
        if not isinstance(metrics, cls):
            raise TypeError(
                "'metrics' should be a `{cls.name}`, a valid list of Metric "
                "instances and/or specs to wrap into one, or None."
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
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """Return a copy of the current state variables.

        This method is designed to expose and share partial results
        that may be aggregated with those of other instances of the
        same metric before computing overall results.

        Returns
        -------
        states: dict[str, dict[str, float or numpy.ndarray]]
            Dict of states that may be fed to another instance of
            this class via its `agg_states` method.
        """
        return {metric.name: metric.get_states() for metric in self.metrics}

    def agg_states(
        self,
        states: Dict[str, Dict[str, Union[float, np.ndarray]]],
    ) -> None:
        """Aggregate provided state variables into self ones.

        This method is designed to aggregate results from multiple
        similar metrics objects into a single one before computing
        its results.

        Parameters
        ----------
        states: dict[str, float or numpy.ndarray]
            Dict of states emitted by another instance of this class
            via its `get_states` method.

        Raises
        ------
        KeyError:
            If any state variable is missing from `states`.
        TypeError:
            If any state variable is of unproper type.
        ValueError:
            If any array state variable is of unproper shape.
        """
        for metric in self.metrics:
            if metric.name in states:
                metric.agg_states(states[metric.name])

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
