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

"""Unit tests for `declearn.metrics.MulticlassAccuracyPrecisionRecall`."""

import os
import sys

import numpy as np
import pytest

from declearn.metrics import MulticlassAccuracyPrecisionRecall

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
    use_scores: bool,
    use_lnames: bool,
) -> MetricTestCase:
    """Return a test case with either predicted labels or scores."""
    # Declare and format the test-case true labels and predictions.
    y_true = [0, 0, 2, 1, 2, 0]
    y_pred = [
        [0.5, 0.2, 0.3],
        [0.7, 0.1, 0.2],
        [0.5, 0.3, 0.2],
        [0.1, 0.6, 0.3],
        [0.3, 0.3, 0.4],
        [0.1, 0.7, 0.2],
    ]
    inputs = {
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
    }
    if not use_scores:
        inputs["y_pred"] = inputs["y_pred"].argmax(axis=1)
    if use_lnames:
        inputs["y_true"] = np.array(["a", "b", "c"])[inputs["y_true"]]
        if not use_scores:
            inputs["y_pred"] = np.array(["a", "b", "c"])[inputs["y_pred"]]
    # Declare the asociated expected metric states and results.
    confmt = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    states = {"confm": confmt}
    scores = {
        "accuracy": 2 / 3,
        "precision": np.array([2 / 3, 1 / 2, 1.0]),
        "recall": np.array([2 / 3, 1.0, 1 / 2]),
        "f-score": np.array([2 / 3, 2 / 3, 2 / 3]),
        "confusion": confmt,
    }
    # Compute expected values of aggregated states and scores.
    agg_states = {"confm": 2 * confmt}
    agg_scores = scores.copy()
    agg_scores["confusion"] = agg_states["confm"]
    # Wrap it all up into a MetricTestCase container.
    metric = MulticlassAccuracyPrecisionRecall(
        labels=["a", "b", "c"] if use_lnames else [0, 1, 2]  # type: ignore
    )
    return MetricTestCase(
        metric, inputs, states, scores, agg_states, agg_scores  # type: ignore
    )


@pytest.mark.parametrize("use_lnames", [False, True], ids=["012", "abc"])
@pytest.mark.parametrize("use_scores", [False, True], ids=["labels", "scores"])
class TestMulticlassAccuracyPrecisionRecall(MetricTestSuite):
    """Unit tests for `MulticlassAccuracyPrecisionRecall`."""

    def test_squeeze(self, test_case: MetricTestCase) -> None:
        with pytest.raises((AssertionError, TypeError)):
            super().test_squeeze(test_case)
