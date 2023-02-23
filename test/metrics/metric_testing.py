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

"""Template test-suite for declearn Metric subclasses."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pytest

from declearn.metrics import Metric
from declearn.test_utils import (
    assert_dict_equal,
    assert_json_serializable_dict,
)


@dataclass
class MetricTestCase:
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
    """

    metric: Metric
    inputs: Dict[str, np.ndarray]
    states: Dict[str, Union[float, np.ndarray]]
    scores: Dict[str, Union[float, np.ndarray]]
    agg_states: Dict[str, Union[float, np.ndarray]]
    agg_scores: Dict[str, Union[float, np.ndarray]]


class MetricTestSuite:
    """Template for declearn Metric subclasses' unit tests suite."""

    tol: Optional[float] = 0.0  # optional tolerance to scores' imperfection

    def test_update(self, test_case: MetricTestCase) -> None:
        """Test that the `update` method works as expected."""
        metric = test_case.metric
        before = metric.get_states()
        metric.update(**test_case.inputs)
        after = metric.get_states()
        assert before.keys() == after.keys()
        with pytest.raises(AssertionError):  # assert not equal
            assert_dict_equal(before, after)
        assert_dict_equal(after, test_case.states)

    def test_zero_results(self, test_case: MetricTestCase) -> None:
        """Test that `get_result` works for un-updated metrics."""
        metric = test_case.metric
        result = metric.get_result()
        assert isinstance(result, dict)
        assert result.keys() == test_case.scores.keys()

    def test_results(self, test_case: MetricTestCase) -> None:
        """Test that the `update` and `get_result` methods work as expected."""
        metric = test_case.metric
        metric.update(**test_case.inputs)
        result = metric.get_result()
        scores = test_case.scores
        assert_dict_equal(result, scores, np_tolerance=self.tol)

    def test_reset(self, test_case: MetricTestCase) -> None:
        """Test that the `reset` method works as expected."""
        metric = test_case.metric
        before = metric.get_states()
        metric.update(**test_case.inputs)
        metric.reset()
        after = metric.get_states()
        assert_dict_equal(before, after)

    def test_aggreg(self, test_case: MetricTestCase) -> None:
        """Test that the `agg_states` method works as expected."""
        # Set up and update two identical metrics.
        metric = test_case.metric
        metbis = deepcopy(test_case.metric)
        metric.update(**test_case.inputs)
        metbis.update(**test_case.inputs)
        # Aggregate the second into the first. Verify that they now differ.
        assert_dict_equal(metric.get_states(), metbis.get_states())
        metbis.agg_states(metric.get_states())
        assert_dict_equal(metric.get_states(), test_case.states)
        with pytest.raises(AssertionError):  # assert not equal
            assert_dict_equal(metric.get_states(), metbis.get_states())
        # Verify the correctness of the aggregated states and scores.
        states = test_case.agg_states
        scores = test_case.agg_scores
        assert_dict_equal(metbis.get_states(), states)
        assert_dict_equal(metbis.get_result(), scores, np_tolerance=self.tol)

    def test_aggreg_errors(self, test_case: MetricTestCase) -> None:
        """Test that the `agg_states` method raises expected exceptions."""
        # Set up a Metric and gather its states after one step.
        metric = test_case.metric
        metric.update(**test_case.inputs)
        states = metric.get_states()
        # Early-exit the test if the Metric has no states.
        if not states:
            return None
        first = list(states)[0]
        # Test that a KeyError is raised if a state is missing from inputs.
        stest = {key: val for key, val in states.items() if key != first}
        with pytest.raises(KeyError):
            metric.agg_states(stest)
        # Test that a TypeError is raised if an input state has wrong type.
        stest = deepcopy(states)
        stest[first] = (
            np.zeros((1, 1)) if isinstance(states[first], float) else 0.0
        )
        with pytest.raises(TypeError):
            metric.agg_states(stest)
        # Test that a ValueError is raised if an input array has wrong shape.
        arrays = [k for k, v in states.items() if isinstance(v, np.ndarray)]
        if arrays:
            stest = deepcopy(states)
            stest[arrays[0]] = np.expand_dims(states[arrays[0]], axis=-1)
            with pytest.raises(ValueError):
                metric.agg_states(stest)
        return None

    def test_squeeze(self, test_case: MetricTestCase) -> None:
        """Test that the metric supports inputs with a squeezable last dim."""
        metric = test_case.metric
        inputs = test_case.inputs
        inpbis = {key: np.expand_dims(val, -1) for key, val in inputs.items()}
        metric.update(**inputs)
        states = metric.get_states()
        metric.reset()
        metric.update(**inpbis)
        st_bis = metric.get_states()
        assert_dict_equal(states, st_bis)

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
