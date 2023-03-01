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

"""Unit and functional tests for the R^2 Metric subclasses."""

import os
import sys
from typing import Dict, Union

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore

from declearn.metrics import RSquared

# dirty trick to import from `metric_testing.py`;
# fmt: off
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metric_testing import MetricTestCase
from test_mae_mse import MeanMetricTestSuite
sys.path.pop()
# pylint: enable=wrong-import-order, wrong-import-position
# fmt: on


@pytest.fixture(name="test_case")
def test_case_fixture(
    weighted: bool,
) -> MetricTestCase:
    """Return a test case for an R2 metric, with opt. sample weights."""
    # Generate random inputs and sample weights.
    np.random.seed(20230301)
    y_true = np.random.normal(scale=1.0, size=32)
    y_pred = y_true + np.random.normal(scale=0.5, size=32)
    s_wght = np.abs(np.random.normal(size=32)) if weighted else np.ones((32,))
    inputs = {"y_true": y_true, "y_pred": y_pred, "s_wght": s_wght}
    # Compute expected intermediate and final results.
    mse = mean_squared_error(y_true, y_pred, sample_weight=s_wght)
    states = {
        "sum_of_squared_errors": s_wght.sum() * mse,
        "sum_of_squared_labels": np.sum(s_wght * np.square(y_true)),
        "sum_of_labels": np.sum(s_wght * y_true),
        "sum_of_weights": s_wght.sum(),
    }
    scores = {
        "r2": r2_score(y_true, y_pred, sample_weight=s_wght)
    }  # type: Dict[str, Union[float, np.ndarray]]
    # Compute derived aggregation results. Wrap as a test case and return.
    agg_states = {key: 2 * val for key, val in states.items()}
    agg_scores = scores.copy()
    metric = RSquared()
    return MetricTestCase(
        metric, inputs, states, scores, agg_states, agg_scores
    )


@pytest.mark.parametrize("weighted", [False, True], ids=["base", "weighted"])
class TestRSquared(MeanMetricTestSuite):
    """Unit tests for `RSquared` Metric."""

    tol = 1e-12  # allow declearn and sklearn scores to differ at 10^-12 prec.

    def test_zero_result(self, test_case: MetricTestCase) -> None:
        """Test that `get_results` works with zero-valued divisor."""
        metric = test_case.metric
        # Case when no samples have been seen: return 0.
        assert metric.get_result() == {metric.name: 0.0}
        # Case when SSt is null but SSr is not: return 0.
        states = getattr(metric, "_states")
        states["sum_of_weights"] = 1.0
        states["sum_of_squared_errors"] = 0.1
        assert metric.get_result() == {metric.name: 0.0}
        # Case when SSt and SSr are null but samples have been seen: return 1.
        states["sum_of_squared_errors"] = 0.0
        assert metric.get_result() == {metric.name: 1.0}
