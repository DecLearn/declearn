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

"""Template test-suite for declearn Metric subclasses."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Generic, Optional, TypeVar, Union
from unittest import mock

import numpy as np
import pytest

from declearn.metrics import Metric, MetricState
from declearn.test_utils import (
    assert_dict_equal,
    assert_json_serializable_dict,
)

MetricStateT = TypeVar("MetricStateT", bound=MetricState)


@dataclass
class MetricTestCase(Generic[MetricStateT]):
    """Base dataclass container for declearn Metric test data.

    Fields
    ------
    metric: Metric
        Base Metric instance on which to conduct unit tests.
    inputs: dict[str, np.ndarray]
        Inputs to the Metric's `update` method.
    states: dict[str, float or np.ndarray]
        Expected metric states after `metric.update(inputs)` has run.
    scores: dict[str, float or np.ndarray]
        Expected metric results after `metric.update(inputs)` has run.
    agg_states: dict[str, float or np.ndarray]
        Expected metric states after aggregating `states` into themselves.
    agg_scores: dict[str, float or np.ndarray]
        Expected metric results after aggregating `states` into themselves.
    supports_secagg: bool
        Whether the Metric is supposed to support SecAgg.
    """

    metric: Metric[MetricStateT]
    inputs: Dict[str, np.ndarray]
    states: MetricStateT
    scores: Dict[str, Union[float, np.ndarray]]
    agg_states: MetricStateT
    agg_scores: Dict[str, Union[float, np.ndarray]]
    supports_secagg: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.states, dict):
            self.states = self.metric.state_cls(**self.states)
        if isinstance(self.agg_states, dict):
            self.agg_states = self.metric.state_cls(**self.agg_states)


class MetricTestSuite:
    """Template for declearn Metric subclasses' unit tests suite."""

    tol: Optional[float] = 0.0  # optional tolerance to scores' imperfection

    def test_build_initial_states(self, test_case: MetricTestCase) -> None:
        """Test that 'build_initial_states' works and returns expected type."""
        metric = test_case.metric
        states = metric.build_initial_states()
        assert isinstance(states, metric.state_cls)

    def test_get_initial_states(self, test_case: MetricTestCase) -> None:
        """Test that initial states can be accessed and match expectations."""
        metric = test_case.metric
        expect = metric.build_initial_states()
        states = metric.get_states()
        assert isinstance(states, metric.state_cls)
        assert_dict_equal(states.to_dict(), expect.to_dict())

    def test_get_initial_results(self, test_case: MetricTestCase) -> None:
        """Test that `get_result` works for un-updated metrics."""
        metric = test_case.metric
        result = metric.get_result()
        assert isinstance(result, dict)
        assert result.keys() == test_case.scores.keys()

    def test_update(self, test_case: MetricTestCase) -> None:
        """Test that the `update` method works as expected."""
        metric = test_case.metric
        before = metric.get_states()
        metric.update(**test_case.inputs)
        after = metric.get_states()
        assert isinstance(after, type(before))
        with pytest.raises(AssertionError):  # assert not equal
            assert_dict_equal(before.to_dict(), after.to_dict())
        assert_dict_equal(after.to_dict(), test_case.states.to_dict())

    def test_get_results_after_update(self, test_case: MetricTestCase) -> None:
        """Test that the `update` and `get_result` methods work as expected."""
        metric = test_case.metric
        metric.update(**test_case.inputs)
        result = metric.get_result()
        scores = test_case.scores
        assert_dict_equal(result, scores, np_tolerance=self.tol)

    def test_set_states(self, test_case: MetricTestCase) -> None:
        """Test that the `set_states` method works as expected."""
        # Instantiate, access initial states, set new ones, access them again.
        metric = test_case.metric
        before = metric.get_states()
        metric.set_states(test_case.agg_states)
        after = metric.get_states()
        # Verify that states were properly set, and differ from initial ones.
        assert after is not test_case.agg_states
        assert_dict_equal(after.to_dict(), test_case.agg_states.to_dict())
        assert after is not before
        with pytest.raises(AssertionError):  # assert not equal
            assert_dict_equal(before.to_dict(), after.to_dict())

    def test_set_states_raises_wrong_input_type(
        self, test_case: MetricTestCase
    ) -> None:
        """Test that the `set_states` method raises on wrong input type."""
        metric = test_case.metric
        with pytest.raises(TypeError):
            metric.set_states(mock.MagicMock())

    def test_reset(self, test_case: MetricTestCase) -> None:
        """Test that the `reset` method works as expected."""
        metric = test_case.metric
        before = metric.get_states()
        metric.update(**test_case.inputs)
        metric.reset()
        after = metric.get_states()
        # Verify that states have the same values, but are distinct objects.
        assert before is not after
        assert_dict_equal(before.to_dict(), after.to_dict())

    def test_states_aggregation(self, test_case: MetricTestCase) -> None:
        """Test that metric states can be aggregated and match expectations."""
        # Set up and update two identical metrics.
        metric = test_case.metric
        metbis = deepcopy(test_case.metric)
        metric.update(**test_case.inputs)
        metbis.update(**test_case.inputs)
        # Gather and aggregate their states. Verify their correctness.
        states = metric.get_states() + metbis.get_states()
        assert isinstance(states, metric.state_cls)
        assert_dict_equal(states.to_dict(), test_case.agg_states.to_dict())

    def test_get_results_after_aggregation(
        self, test_case: MetricTestCase
    ) -> None:
        """Test that 'get_results' returns expected values from aggregation."""
        # Set up and update two identical metrics.
        metric = test_case.metric
        metbis = deepcopy(test_case.metric)
        metric.update(**test_case.inputs)
        metbis.update(**test_case.inputs)
        # Aggregate their states and assign into the first metric.
        metric.set_states(metric.get_states() + metbis.get_states())
        # Verify that the resulting values match expectations.
        expect = test_case.agg_scores
        assert_dict_equal(metric.get_result(), expect, np_tolerance=self.tol)

    def test_update_with_squeezable_inputs(
        self, test_case: MetricTestCase
    ) -> None:
        """Test that the metric supports inputs with a squeezable last dim."""
        # Set up a Metric and two identical inputs up to squeezable last dim.
        metric = test_case.metric
        inputs = test_case.inputs
        inpbis = {key: np.expand_dims(val, -1) for key, val in inputs.items()}
        # Gather the states resulting from either inputs being processed.
        metric.update(**inputs)
        states = metric.get_states()
        metric.reset()
        metric.update(**inpbis)
        st_bis = metric.get_states()
        # Verify that they are identical.
        assert_dict_equal(states.to_dict(), st_bis.to_dict())

    def test_config(self, test_case: MetricTestCase) -> None:
        """Test that the metric supports (de)serialization from a dict."""
        metric = test_case.metric
        # Test that `get_config` returns a JSON-serializable dict.
        config = metric.get_config()
        assert_json_serializable_dict(config)
        # Test that `from_config` produces a similar Metric.
        metbis = type(metric).from_config(config)
        assert isinstance(metbis, type(metric))
        cfgbis = metbis.get_config()
        assert_dict_equal(cfgbis, config)
        # Test that `from_specs` works properly as well.
        metter = Metric.from_specs(metric.name, config)
        assert isinstance(metter, type(metric))
        cfgter = metter.get_config()
        assert_dict_equal(cfgter, config)

    def test_secagg_compatibility(self, test_case: MetricTestCase) -> None:
        """Test that the metric supports secure aggregation."""
        # Set up a Metric and gather its states after one update.
        metric = test_case.metric
        metric.update(**test_case.inputs)
        states = metric.get_states()
        # If SecAgg is expected not be supported, exit after catching error.
        if not test_case.supports_secagg:
            with pytest.raises(NotImplementedError):
                states.prepare_for_secagg()
            return
        # Otherwise, call 'prepare_for_secagg' and verify return types.
        secagg, clrtxt = states.prepare_for_secagg()
        assert isinstance(secagg, dict)
        assert isinstance(clrtxt, dict) or (clrtxt is None)
        # Perform aggregation as defined for SecAgg (but in cleartext).
        secagg = {key: val + val for key, val in secagg.items()}
        if clrtxt is None:
            clrtxt = {}
        else:
            clrtxt = {
                key: getattr(
                    states, f"aggregate_{key}", states.default_aggregate
                )(val, val)
                for key, val in clrtxt.items()
            }
        output = type(states)(**secagg, **clrtxt)
        # Verify that results match expectations.
        expect = states + states
        assert_dict_equal(expect.to_dict(), output.to_dict())
