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

"""Unit tests for `declearn.metrics.BinaryRocAUC`."""

import os
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np
import pytest
import sklearn  # type: ignore

from declearn.metrics import BinaryRocAUC
from declearn.test_utils import make_importable

# relative imports from `metric_testing.py`
with make_importable(os.path.dirname(__file__)):
    from metric_testing import MetricTestCase, MetricTestSuite


@pytest.fixture(name="test_case")
def test_case_fixture(
    case: Literal["1d", "2d"],
    scale: float,  # Literal[0.1, 0.2]
    bound: bool,  # (0.2, 0.8) if True else None
) -> MetricTestCase:
    """Return a test case with either 1-D or 2-D samples."""
    # Fetch pre-computed inputs and resulting states and scores.
    inputs, states, scores = (
        _test_case_1d() if case == "1d" else _test_case_2d()
    )
    # Adjust expected parameter when using non-default bound parameter.
    if bound:
        states = {
            key: val[2:-2] if isinstance(val, np.ndarray) else val
            for key, val in states.items()
        }
        scores = {
            key: val[2:-2] if isinstance(val, np.ndarray) else val
            for key, val in scores.items()
        }
        scores["roc_auc"] = sklearn.metrics.auc(scores["fpr"], scores["tpr"])
    # Adjust expected results when using non-default scale parameter.
    if scale == 0.2:
        states = {
            key: val[::2] if isinstance(val, np.ndarray) else val
            for key, val in states.items()
        }
        scores = {
            key: val[::2] if isinstance(val, np.ndarray) else val
            for key, val in scores.items()
        }
    elif scale != 0.1:
        raise ValueError("Unsupported 'scale' testing parameter.")
    # Compute expected aggregated states and scores.
    agg_states: Dict[str, Any] = {key: 2 * val for key, val in states.items()}
    agg_states["thresh"] = states["thresh"]
    agg_scores = scores.copy()
    # Instantiate a BinaryRocAUC and return a MetricTestCase.
    metric = BinaryRocAUC(scale=scale, bound=((0.2, 0.8) if bound else None))
    supports_secagg = bound
    return MetricTestCase(
        metric, inputs, states, scores, agg_states, agg_scores, supports_secagg
    )


def _test_case_1d() -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, Union[float, np.ndarray]],
    Dict[str, Union[float, np.ndarray]],
]:
    """Return a test case with 1-D samples (standard binary classif)."""
    # similar inputs as for Binary APR; pylint: disable=duplicate-code
    inputs = {
        "y_true": np.array([0, 0, 1, 1]),
        "y_pred": np.array([4, 8, 6, 8]) / 10,
    }
    # pylint: enable=duplicate-code
    states: Dict[str, Union[float, np.ndarray]] = {
        "tpos": np.array(
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0]
        ),
        "tneg": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
        ),
        "fpos": np.array(
            [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        ),
        "fneg": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        ),
        "thresh": np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
    }
    scores: Dict[str, Union[float, np.ndarray]] = {
        "tpr": np.array(
            [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ),
        "fpr": np.array(
            [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
        ),
        "thresh": np.array(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        ),
        "roc_auc": 0.625,
    }
    return inputs, states, scores


def _test_case_2d() -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, Union[float, np.ndarray]],
    Dict[str, Union[float, np.ndarray]],
]:
    """Return a test case with 2-D samples (multilabel binary classif)."""
    # similar inputs as for Binary APR; pylint: disable=duplicate-code
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
    # pylint: enable=duplicate-code
    states: Dict[str, Union[float, np.ndarray]] = {
        "tpos": np.array(
            [6.0, 6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 3.0, 3.0, 2.0, 0.0]
        ),
        "tneg": np.array(
            [0.0, 0.0, 2.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]
        ),
        "fpos": np.array(
            [6.0, 6.0, 4.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "fneg": np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 3.0, 3.0, 4.0, 6.0]
        ),
        "thresh": np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
    }
    scores: Dict[str, Union[float, np.ndarray]] = {
        "tpr": np.array(
            [0.0, 1 / 3, 0.5, 0.5, 2 / 3, 2 / 3, 1.0, 1.0, 1.0, 1.0, 1.0]
        ),
        "fpr": np.array(
            [0.0, 0.0, 0.0, 0.0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 2 / 3, 1.0, 1.0]
        ),
        "thresh": np.array(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        ),
        "roc_auc": 0.9305555555555556,
    }
    return inputs, states, scores


@pytest.mark.parametrize("bound", [False, True], ids=["unbound", "bound"])
@pytest.mark.parametrize("scale", [0.1, 0.2])
@pytest.mark.parametrize("case", ["1d", "2d"])
class TestBinaryRocAUC(MetricTestSuite):
    """Unit tests for `BinaryRocAUC`."""

    def test_aggreg_unaligned_thresholds(self, test_case):
        """Test the aggregation of states aligned to distinct thresholds."""
        metric = test_case.metric
        metbis = BinaryRocAUC(scale=0.05, bound=(0.4, 1.2))
        metric.update(**test_case.inputs)
        metbis.update(**test_case.inputs)
        # Test that bound into bound fails as boundaries differ.
        if metric.bound:
            with pytest.raises(TypeError):
                metric.set_states(metbis.get_states())
            with pytest.raises(TypeError):
                metbis.set_states(metric.get_states())
        else:
            # Gather states from both metrics.
            unbound = metric.get_states()
            bounded = metbis.get_states()
            assert type(bounded)
            # Test that bound + unbound fails, as it would change type.
            with pytest.raises(ValueError):
                bounded + unbound  # pylint: disable=pointless-statement
            # Test that assignment of unbound states into bounded metric fails.
            with pytest.raises(TypeError):
                metbis.set_states(unbound)
            # Test that unbound + bound works, with thresholds being updated.
            aggregated = unbound + bounded
            metric.set_states(aggregated)
            assert metric.bound is None
            thresh = metric.get_states().thresh
            thrbis = unbound.thresh
            assert all(val in thresh for val in thrbis)
            # Test that assignment of bounded into unbound metric works.
            metric.set_states(bounded)
            assert isinstance(metric.get_states(), type(unbound))
