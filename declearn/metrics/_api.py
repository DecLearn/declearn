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

"""Iterative and federative evaluation metrics base class."""

import abc
from copy import deepcopy
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Optional,
    Self,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from declearn.utils import (
    Aggregate,
    access_registered,
    create_types_registry,
    register_type,
)

__all__ = [
    "Metric",
    "MetricState",
]


class MetricState(
    Aggregate, base_cls=True, register=False, metaclass=abc.ABCMeta
):
    """Abstract base class for Metrics intermediate aggregatable states.

    Each and every `Metric` subclass is expected to be coupled with one
    (or multiple) `MetricState` subtypes, which are used to exchange and
    aggregate partial results across a network of peers, which can in the
    end be passed to a single `Metric` instance for metrics' finalization.

    This class also defines whether contents are compatible with secure
    aggregation, and whether some fields should remain in cleartext no
    matter what.

    Note that subclasses are automatically type-registered, and should be
    decorated as `dataclasses.dataclass`. To prevent registration, simply
    pass `register=False` at inheritance.
    """

    _group_key = "MetricState"


MetricStateT = TypeVar("MetricStateT", bound=MetricState)


@create_types_registry(name="Metric")
class Metric(Generic[MetricStateT], metaclass=abc.ABCMeta):
    """Abstract class defining an API to compute federative metrics.

    This class defines an API to instantiate stateful containers
    for one or multiple metrics, that enable computing the final
    results through iterative update steps that may additionally
    be run in a federative way.

    Usage
    -----
    Single-party usage:
    ```
    >>> metric = MetricSubclass()
    >>> metric.update(y_true, y_pred)  # take one update state
    >>> metric.get_result()    # after one or multiple updates
    >>> metric.reset()  # reset before a next evaluation round
    ```

    Multiple-parties usage:
    ```
    >>> # Instantiate 2+ metrics and run local update steps.
    >>> metric_0 = MetricSubclass()
    >>> metric_1 = MetricSubclass()
    >>> metric_0.udpate(y_true_0, y_pred_0)
    >>> metric_1.update(y_true_1, y_pred_1)
    >>> # Gather and share metric states (aggregated information).
    >>> states_0 = metric_0.get_states()  # metric_0 is unaltered
    >>> states_1 = metric_1.get_states()  # metric_1 is unaltered
    >>> # Compute results that aggregate info from both clients.
    >>> states = states_0 + states_1
    >>> metric_0.set_states(states)  # would work the same with metrics_1
    >>> metric_0.get_result()
    ```

    Abstract
    --------
    To define a concrete Metric, one must subclass it and define:

    - name: str class attribute
        Name identifier of the class (should be unique across existing
        Metric classes). Also used for automatic types-registration of
        the class (see `Inheritance` section below).
    - build_initial_states() -> MetricState:
        Return the initial states for this Metric instance.
        This method is called to initialize the `_states` attribute,
        that should be used and updated by other abstract methods.
    - update(y_true: np.ndarray, y_pred: np.ndarray, s_wght: np.ndarray|None):
        Update the metric's internal state based on a data batch.
        This method should update `self._states` in-place.
    - get_result() -> dict[str, (float | np.ndarray)]:
        Compute the metric(s), based on the current state variables.
        This method should make use of `self._states` and prevent
        side effects on its contents.

    Overridable
    -----------
    Some methods may be overridden based on the concrete Metric's needs:

    - reset():
        Reset the metric to its initial state.
    - get_states() -> MetricState:
        Return a copy of the current state variables.
    - set_states(MetricState):
        Replace current state variables with a copy of inputs.

    Finally, depending on the hyper-parameters defined by the subclass's
    `__init__`, one should adjust JSON-configuration-interfacing methods:

    - get_config() -> dict[str, any]:
        Return a JSON-serializable configuration dict for this Metric.
    - from_config(config: dict[str, any]) -> Self:
        Instantiate a Metric from its configuration dict.

    Inheritance
    -----------
    When a subclass inheriting from `Metric` is declared, it is automatically
    registered under the "Metric" group using its class-attribute `name`.
    This can be prevented by adding `register=False` to the inheritance specs
    (e.g. `class MyCls(Metric, register=False)`).

    See `declearn.utils.register_type` for details on types registration.
    """

    name: ClassVar[str]
    """Name identifier of the class, unique across Metric classes."""

    state_cls: ClassVar[Type[MetricState]]
    """Type of 'MetricState' data structure used by this 'Metric' class."""

    def __init__(
        self,
    ) -> None:
        """Instantiate the metric object."""
        self._states = self.build_initial_states()

    @abc.abstractmethod
    def build_initial_states(
        self,
    ) -> MetricStateT:
        """Return the initial states for this Metric instance.

        Returns
        -------
        states:
            Initial internal states for this object, as a `MetricState`.
        """

    @abc.abstractmethod
    def get_result(
        self,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Compute finalized metric(s), based on the current state variables.

        Returns
        -------
        results: dict[str, float or numpy.ndarray]
            Dict of named result metrics, that may either be
            unitary float scores or numpy arrays.
        """

    @abc.abstractmethod
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

    def reset(
        self,
    ) -> None:
        """Reset the metric to its initial state."""
        self._states = self.build_initial_states()

    def get_states(
        self,
    ) -> MetricStateT:
        """Return a copy of the current state variables.

        This method is designed to expose and share partial results
        that may be aggregated with those of other instances of the
        same metric before computing overall results.

        Returns
        -------
        states:
            Copy of current states, as a `MetricState` instance.
        """
        return deepcopy(self._states)

    def set_states(
        self,
        states: MetricStateT,
    ) -> None:
        """Replace internal states with a copy of incoming ones.

        Parameters
        ----------
        states:
            Replacement states, as a compatible `MetricState` instance.

        Raises
        ------
        TypeError
            If `states` is of improper type.
        """
        if not isinstance(states, self.state_cls):
            raise TypeError(
                f"'{self.__class__.__name__}.set_states' expected "
                f"'{self.state_cls}' inputs, got '{type(states)}'."
            )
        self._states = deepcopy(states)  # type: ignore

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
        KeyError
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

    @staticmethod
    def _prepare_sample_weights(
        s_wght: Optional[np.ndarray],
        n_samples: int,
    ) -> np.ndarray:
        """Flatten or generate sample weights and validate their shape.

        This method is a shared util that may or may not be used as part
        of concrete Metric classes' backend depending on their formula.

        Parameters
        ----------
        s_wght: np.ndarray or None
            1-d (or squeezable) array of sample-wise positive scalar
            weights. If None, one will be generated, with one values.
        n_samples: int
            Expected length of the sample weights.

        Returns
        -------
        s_wght: np.ndarray
            Input (opt. squeezed) `s_wght`, or `np.ones(n_samples)`
            if input was None.

        Raises
        ------
        ValueError
            If the input array has improper shape or negative values.
        """
        if s_wght is None:
            return np.ones(shape=(n_samples,))
        s_wght = s_wght.squeeze()
        if s_wght.shape != (n_samples,) or np.any(s_wght < 0):
            raise ValueError(
                "Improper shape for 's_wght': should be a 1-d array "
                "of sample-wise positive scalar weights."
            )
        return s_wght
