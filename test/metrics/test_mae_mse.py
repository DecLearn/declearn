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

"""Unit and functional tests for the MAE and MSE Metric subclasses."""

import os
import sys
from typing import Dict, Literal, Union

import numpy as np
import pytest

from declearn.metrics import MeanAbsoluteError, MeanSquaredError, Metric


# dirty trick to import from `metric_testing.py`;
# fmt: off
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metric_testing import MetricTestCase, MetricTestSuite
sys.path.pop()
# pylint: enable=wrong-import-order, wrong-import-position
# fmt: on


@pytest.fixture(name="test_case")
def test_case_fixture(
    case: Literal["mae", "mse"],
    weighted: bool,
) -> MetricTestCase:
    """Return a test case for a MAE or MSE metric, with opt. sample weights."""
    # Generate random inputs and compute the expected sum of errors.
    y_true = np.zeros((32,))
    y_pred = np.random.normal(size=(32,))
    inputs = {"y_true": y_true, "y_pred": y_pred}
    if case == "mae":
        metric = MeanAbsoluteError()  # type: Metric
        errors = np.abs(y_pred)
    else:
        metric = MeanSquaredError()
        errors = np.square(y_pred)
    # Compute expected results, optionally using sample weights.
    if weighted:
        s_wght = inputs["s_wght"] = np.abs(np.random.normal(size=(32,)))
        errors = errors * s_wght
        states = {"current": errors.sum(), "divisor": s_wght.sum()}
    else:
        states = {"current": errors.sum(), "divisor": 32}
    scores = {
        case: states["current"] / states["divisor"]
    }  # type: Dict[str, Union[float, np.ndarray]]
    # Compute derived aggregation results. Wrap as a test case and return.
    agg_states = {key: 2 * val for key, val in states.items()}
    agg_scores = scores.copy()
    return MetricTestCase(
        metric, inputs, states, scores, agg_states, agg_scores
    )


class MeanMetricTestSuite(MetricTestSuite):
    """Unit tests suite for `MeanMetric` subclasses."""

    def test_update_errors(self, test_case: MetricTestCase) -> None:
        """Test that `update` raises on improper `s_wght` shapes."""
        metric = test_case.metric
        inputs = test_case.inputs
        # Test with multi-dimensional sample weights.
        s_wght = np.ones(shape=(len(inputs["y_pred"]), 2))
        with pytest.raises(ValueError):
            metric.update(inputs["y_true"], inputs["y_pred"], s_wght)
        # Test with improper-length sample weights.
        s_wght = np.ones(shape=(len(inputs["y_pred"]) + 2,))
        with pytest.raises(ValueError):
            metric.update(inputs["y_true"], inputs["y_pred"], s_wght)

    def test_zero_result(self, test_case: MetricTestCase) -> None:
        """Test that `get_results` works with zero-valued divisor."""
        metric = test_case.metric
        assert metric.get_result() == {metric.name: 0.0}


@pytest.mark.parametrize("weighted", [False, True], ids=["base", "weighted"])
@pytest.mark.parametrize("case", ["mae"])
class TestMeanAbsoluteError(MeanMetricTestSuite):
    """Unit tests for `MeanAbsoluteError`."""


@pytest.mark.parametrize("weighted", [False, True], ids=["base", "weighted"])
@pytest.mark.parametrize("case", ["mse"])
class TestMeanSquaredError(MeanMetricTestSuite):
    """Unit tests for `MeanSquaredError`."""
