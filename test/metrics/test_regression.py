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

"""Unit and functional tests for the R2 Metric subclasses."""

import os
import sys
from typing import Dict, Literal, Union

import numpy as np
import pytest

from declearn.metrics import R2

# HACK to import from `metric_testing.py`;
# fmt: off
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metric_testing import MetricTestCase, MetricTestSuite

sys.path.pop()
# pylint: enable=wrong-import-order, wrong-import-position


@pytest.fixture(name="test_case")
def test_case_fixture(
    weighted: bool,
    # n_dims: int,
) -> MetricTestCase:
    """Return a test case for an R2 metric, with opt. sample weights."""
    # Generate random inputs and compute the expected sum of errors.
    y_true = np.concatenate((np.ones((16,)),np.ones((16,))*-1))
    y_pred = np.random.normal(size=(32,))
    inputs = {"y_true": y_true, "y_pred": y_pred}
    metric = R2()
    inputs["s_wght"] = np.ones((32,))
    wght = 1.0
    if weighted:
        inputs["s_wght"] *= 1.5
        wght *= 1.5
    # Compute expected results, optionally using sample weights.
    exp_var = (wght * (y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
    states = {
            "explained_variance": exp_var,
            "weighted_sum_of_sq": 32.0 * wght,
            "weighted_sample_count": 32.0 * wght,
            "weighted_sum": 0.0 * wght,
        }
    true_var = (
            states["weighted_sum_of_sq"]
            - states["weighted_sum"] ** 2
            / states["weighted_sample_count"]
        )
    scores = {
        'r2': 1 - states["explained_variance"] / true_var
    }  
    # type: Dict[str, Union[float, np.ndarray]]
    # Compute derived aggregation results. Wrap as a test case and return.
    agg_states = {key: 2 * val for key, val in states.items()}
    agg_scores = scores.copy()
    return MetricTestCase(
        metric, inputs, states, scores, agg_states, agg_scores
    )


class R2TestSuite(MetricTestSuite):
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
        assert metric.get_result() == {metric.name: 1.0}
        metric._states['explained_variance'] = 0.1
        assert metric.get_result() == {metric.name: 0.0}


@pytest.mark.parametrize("weighted", [False, True], ids=["base", "weighted"])
class TestR2(R2TestSuite):
    """Unit tests for `MeanAbsoluteError`."""
