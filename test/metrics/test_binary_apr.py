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

"""Unit tests for `declearn.metrics.BinaryAccuracyPrecisionRecall`."""

import os
import sys
from typing import Dict, Literal, Union, Tuple

import numpy as np
import pytest

from declearn.metrics import BinaryAccuracyPrecisionRecall

# dirty trick to import from `metric_testing.py`;
# fmt: off
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metric_testing import MetricTestCase, MetricTestSuite
sys.path.pop()
# pylint: enable=wrong-import-order, wrong-import-position


@pytest.fixture(name="test_case")
def test_case_fixture(
    case: Literal["1d", "2d"],
    thresh: float,
) -> MetricTestCase:
    """Return a test case with either 1-D or 2-D samples for a threshold."""
    inputs, states, scores = (
        _test_case_1d(thresh) if case == "1d" else _test_case_2d(thresh)
    )
    # Add the F1-score to expected scores.
    scores["f-score"] = (
        (states["tpos"] + states["tpos"])
        / (states["tpos"] + states["tpos"] + states["fpos"] + states["fneg"])
    )
    # Add the confusion matrix to expected scores.
    confmt = [
        [states["tneg"], states["fpos"]],
        [states["fneg"], states["tpos"]],
    ]
    scores["confusion"] = np.array(confmt)
    # Compute expected values of aggregated states and scores.
    agg_states = {key: 2 * val for key, val in states.items()}
    agg_scores = scores.copy()
    agg_scores["confusion"] = 2 * scores["confusion"]
    # Wrap it all up into a MetricTestCase container.
    metric = BinaryAccuracyPrecisionRecall(thresh=thresh)
    return MetricTestCase(
        metric, inputs, states, scores, agg_states, agg_scores
    )


def _test_case_1d(
    thresh: float,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, Union[float, np.ndarray]],
    Dict[str, Union[float, np.ndarray]],
]:
    """Return a test case with 1-D samples (standard binary classif)."""
    inputs = {
        "y_true": np.array([0, 0, 1, 1]),
        "y_pred": np.array([4, 8, 6, 8]) / 10,
    }
    states = {
        0.3: {"tpos": 2.0, "tneg": 0.0, "fpos": 2.0, "fneg": 0.0},
        0.5: {"tpos": 2.0, "tneg": 1.0, "fpos": 1.0, "fneg": 0.0},
        0.7: {"tpos": 1.0, "tneg": 1.0, "fpos": 1.0, "fneg": 1.0},
    }[
        thresh
    ]
    scores = {
        0.3: {"accuracy": 2 / 4, "precision": 2 / 4, "recall": 2 / 2},
        0.5: {"accuracy": 3 / 4, "precision": 2 / 3, "recall": 2 / 2},
        0.7: {"accuracy": 2 / 4, "precision": 1 / 2, "recall": 1 / 2},
    }[
        thresh
    ]
    return inputs, states, scores  # type: ignore


def _test_case_2d(
    thresh: float,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, Union[float, np.ndarray]],
    Dict[str, Union[float, np.ndarray]],
]:
    """Return a test case with 2-D samples (multilabel binary classif)."""
    inputs = {
        "y_true": np.array(
            [
                [0, 0, 1, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
            ]
        ),
        "y_pred": 0.1
        * np.array(
            [
                [1, 2, 8, 9],
                [6, 4, 2, 1],
                [6, 4, 9, 2],
            ]
        ),
    }
    states = {
        0.3: {"tpos": 6.0, "tneg": 5.0, "fpos": 1.0, "fneg": 0.0},
        0.5: {"tpos": 4.0, "tneg": 5.0, "fpos": 1.0, "fneg": 2.0},
        0.7: {"tpos": 3.0, "tneg": 6.0, "fpos": 0.0, "fneg": 3.0},
    }[
        thresh
    ]
    scores = {
        0.3: {"accuracy": 11 / 12, "precision": 6 / 7, "recall": 6 / 6},
        0.5: {"accuracy": 9 / 12, "precision": 4 / 5, "recall": 4 / 6},
        0.7: {"accuracy": 9 / 12, "precision": 3 / 3, "recall": 3 / 6},
    }[
        thresh
    ]
    return inputs, states, scores  # type: ignore


@pytest.mark.parametrize("thresh", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("case", ["1d", "2d"])
class TestBinaryAccuracyPrecisionRecall(MetricTestSuite):
    """Unit tests for `BinaryAccuracyPrecisionRecall`."""
