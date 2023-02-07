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

"""Iterative and federative evaluation metrics base class."""

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, ClassVar, Dict, Optional, Union

import numpy as np
from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.utils import (
    access_registered,
    create_types_registry,
    register_type,
)

__all__ = [
    "Metric",
]


@create_types_registry(name="Metric")
class Metric(metaclass=ABCMeta):
    """Abstract class defining an API to compute federative metrics.

    This class defines an API to instantiate stateful containers
    for one or multiple metrics, that enable computing the final
    results through iterative update steps that may additionally
    be run in a federative way.

    Single-party usage:
    >>> metric = MetricSubclass()
    >>> metric.update(y_true, y_pred)  # take one update state
    >>> metric.get_result()    # after one or multiple updates
    >>> metric.reset()  # reset before a next evaluation round

    Multiple-parties usage:
    >>> # Instantiate 2+ metrics and run local update steps.
    >>> metric_0 = MetricSubclass()
    >>> metric_1 = MetricSubclass()
    >>> metric_0.udpate(y_true_0, y_pred_0)
    >>> metric_1.update(y_true_1, y_pred_1)
    >>> # Gather and share metric states (aggregated information).
    >>> states_0 = metric_0.get_states()  # metrics_0 is unaltered
    >>> metric_1.agg_states(states_0)     # metrics_1 is updated
    >>> # Compute results that aggregate info from both clients.
    >>> metric_1.get_result()

    Abstract
    --------
    To define a concrete Metric, one must subclass it and define:

    name: str class attribute
        Name identifier of the class (should be unique across existing
        Metric classes). Also used for automatic types-registration of
        the class (see `Inheritance` section below).
    _build_states() -> dict[str, (float | np.ndarray)]:
        Build and return an ensemble of state variables.
        This method is called to initialize the `_states` attribute,
        that should be used and updated by other abstract methods.
    update(y_true: np.ndarray, y_pred: np.ndarray, s_wght: (np.ndarray|None)):
        Update the metric's internal state based on a data batch.
        This method should update `self._states` in-place.
    get_result() -> dict[str, (float | np.ndarray)]:
        Compute the metric(s), based on the current state variables.
        This method should make use of `self._states` and prevent
        side effects on its contents.

    Overridable
    -----------
    Some methods may be overridden based on the concrete Metric's needs.
    The most imporant one is the states-aggregation method:

    agg_states(states: dict[str, (float | np.ndarray)]:
        Aggregate provided state variables into self ones.
        By default, it expects input and internal states to have
        similar specifications, and aggregates them by summation,
        which might no be proper depending on the actual metric.

    A pair of methods may be extended to cover non-`self._states`-contained
    variables:

    reset():
        Reset the metric to its initial state.
    get_states() -> dict[str, (float | np.ndarray)]:
        Return a copy of the current state variables.


    Finally, depending on the hyper-parameters defined by the subclass's
    `__init__`, one should adjust JSON-configuration-interfacing methods:

    get_config() -> dict[str, any]:
        Return a JSON-serializable configuration dict for this Metric.
    from_config(config: dict[str, any]) -> Self:
        Instantiate a Metric from its configuration dict.

    Inheritance
    -----------
    When a subclass inheriting from `Metric` is declared, it is automatically
    registered under the "Metric" group using its class-attribute `name`.
    This can be prevented by adding `register=False` to the inheritance specs
    (e.g. `class MyCls(Metric, register=False)`).

    See `declearn.utils.register_type` for details on types registration.
    """

    name: ClassVar[str] = NotImplemented

    def __init__(
        self,
    ) -> None:
        """Instantiate the metric object."""
        self._states = self._build_states()

    @abstractmethod
    def _build_states(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Build and return an ensemble of state variables.

        The state variables stored in this dict are (by default)
        sharable with other instances of this metric and may be
        combined with the latter's through summation in order to
        compute final metrics in a federated way.

        Note that the update process may be altered by extending
        or overridding the `agg_states` method.

        Returns
        -------
        states: dict[str, float or numpy.ndarray]
            Dict of initial states that are to be assigned as
            `_states` private attribute.
        """

    @abstractmethod
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

    @abstractmethod
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

    @staticmethod
    def normalize_weights(s_wght: np.ndarray) -> np.ndarray:
        """Utility method to ensure weights sum to one.

        Note that this method may or may not be used depending on
        the actual `Metric` considered, and is merely provided as
        a utility to metric developers.
        """
        if s_wght.sum():
            s_wght /= s_wght.sum()
        else:
            raise ValueError(
                "Weights provided sum to zero, please provide only "
                "positive weights with at least one non-zero weight."
            )
        return s_wght

    def reset(
        self,
    ) -> None:
        """Reset the metric to its initial state."""
        self._states = self._build_states()

    def get_states(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Return a copy of the current state variables.

        This method is designed to expose and share partial results
        that may be aggregated with those of other instances of the
        same metric before computing overall results.

        Returns
        -------
        states: dict[str, float or numpy.ndarray]
            Dict of states that may be fed to another instance of
            this class via its `agg_states` method.
        """
        return deepcopy(self._states)

    def agg_states(
        self,
        states: Dict[str, Union[float, np.ndarray]],
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
        final = {}  # type: Dict[str, Union[float, np.ndarray]]
        # Iteratively compute sum-aggregated states, running sanity checks.
        for name, own in self._states.items():
            if name not in states:
                raise KeyError(f"Missing required state variable: '{name}'.")
            oth = states[name]
            if not isinstance(oth, type(own)):
                raise TypeError(f"Input state '{name}' is of unproper type.")
            if isinstance(own, np.ndarray):
                if own.shape != oth.shape:  # type: ignore
                    msg = f"Input state '{name}' is of unproper shape."
                    raise ValueError(msg)
            final[name] = own + oth
        # Assign the sum-aggregated states.
        self._states = final

    def __init_subclass__(
        cls,
        register: bool = True,
        **kwargs: Any,
    ) -> None:
        """Automatically type-register Metric subclasses."""
        super().__init_subclass__(**kwargs)
        if register:
            register_type(cls, name=cls.name, group="Metric")

    def get_config(
        self,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable configuration dict for this Metric."""
        return {}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> Self:
        """Instantiate a Metric from its configuration dict."""
        return cls(**config)

    @staticmethod
    def from_specs(
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> "Metric":
        """Instantiate a Metric from its registered name and config dict.

        Parameters
        ----------
        name: str
            Name based on which the metric can be retrieved.
            Available as a class attribute.
        config: dict[str, any] or None
            Configuration dict of the metric, that is to be
            passed to its `from_config` class constructor.

        Raises
        ------
        KeyError:
            If the provided `name` fails to be mapped to a registered
            Metric subclass.
        """
        try:
            cls = access_registered(name, group="Metric")
        except KeyError as exc:
            raise KeyError(
                f"Failed to retrieve Metric subclass from name '{name}'."
            ) from exc
        return cls.from_config(config or {})
