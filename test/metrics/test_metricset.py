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

"""Unit tests for `declearn.metrics.MetricSet`."""

from typing import Tuple
from unittest import mock

import numpy as np
import pytest

from declearn.metrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricSet,
    MetricState,
)


def get_mock_metricset() -> Tuple[
    MeanAbsoluteError, MeanSquaredError, MetricSet
]:
    """Provide with a MetricSet wrapping mock metrics."""
    mae = mock.create_autospec(MeanAbsoluteError, instance=True)
    mae.name = MeanAbsoluteError.name
    mse = mock.create_autospec(MeanSquaredError, instance=True)
    mse.name = MeanSquaredError.name
    metrics = MetricSet([mae, mse])
    return mae, mse, metrics


class TestMetricSet:
    """Unit tests for `declearn.metrics.MetricSet`."""

    def test_init_by_object(self) -> None:
        """Test that instanciation with a pre-instantiated Metric works."""
        metric = MeanAbsoluteError()
        metrics = MetricSet([metric])
        assert metrics.metrics == [metric]

    def test_init_by_name(self) -> None:
        """Test that instanciation with a Metric's identifier name works."""
        metrics = MetricSet(["mae"])
        assert isinstance(metrics.metrics, list) and len(metrics.metrics) == 1
        assert isinstance(metrics.metrics[0], MeanAbsoluteError)

    def test_init_by_specs(self) -> None:
        """Test that instanciation with a Metric's specs works."""
        metrics = MetricSet([("mae", {})])
        assert isinstance(metrics.metrics, list) and len(metrics.metrics) == 1
        assert isinstance(metrics.metrics[0], MeanAbsoluteError)

    def test_init_errors(self) -> None:
        """Test that `MetricSet.__init__` documented exceptions are raised."""
        # Test that wrapping multiple instances of the same Metric fails.
        mae_a = MeanAbsoluteError()
        mae_b = MeanAbsoluteError()
        with pytest.raises(RuntimeError):
            MetricSet([mae_a, mae_b])
        # Test that providing with a non-Metric instance raises a TypeError.
        wrong_inputs = [{0: "mae"}]
        with pytest.raises(TypeError):
            MetricSet(wrong_inputs)  # type: ignore  # voluntary mistake
        # Test that providing with unmapped identifier raises a KeyError.
        with pytest.raises(KeyError):
            MetricSet([f"random-name-{np.random.uniform()}"])

    def test_get_result(self) -> None:
        """Test that `MetricSet.get_result` works as expected."""
        mae, mse, metrics = get_mock_metricset()
        results = metrics.get_result()
        assert isinstance(results, dict)
        mae.get_result.assert_called_once()  # type: ignore  # mock
        mse.get_result.assert_called_once()  # type: ignore  # mock

    def test_update(self) -> None:
        """Test that `MetricSet.update` works as expected."""
        mae, mse, metrics = get_mock_metricset()
        inputs = {
            "y_true": np.random.normal((8, 32)),
            "y_pred": np.random.normal((8, 32)),
            "s_wght": None,
        }
        metrics.update(**inputs)  # type: ignore
        mae.update.assert_called_once_with(**inputs)  # type: ignore  # mock
        mse.update.assert_called_once_with(**inputs)  # type: ignore  # mock

    def test_reset(self) -> None:
        """Test that `MetricSet.reset` works as expected."""
        mae, mse, metrics = get_mock_metricset()
        metrics.reset()
        mae.reset.assert_called_once()  # type: ignore  # mock
        mse.reset.assert_called_once()  # type: ignore  # mock

    def test_get_states(self) -> None:
        """Test that `MetricSet.get_states` works as expected."""
        mae, mse, metrics = get_mock_metricset()
        states = metrics.get_states()
        assert isinstance(states, dict)
        mae.get_states.assert_called_once()  # type: ignore  # mock
        mse.get_states.assert_called_once()  # type: ignore  # mock

    def test_set_states(self) -> None:
        """Test that `MetricSet.set_states` works as expected."""
        mae, mse, metrics = get_mock_metricset()
        states = {
            "mae": mock.create_autospec(MetricState, instance=True),
            "mse": mock.create_autospec(MetricState, instance=True),
        }
        metrics.set_states(states)
        mae.set_states.assert_called_once_with(  # type: ignore  # mock
            states["mae"]
        )
        mse.set_states.assert_called_once_with(  # type: ignore  # mock
            states["mse"]
        )

    def test_get_config(self) -> None:
        """Test that `MetricSet.get_config` works as expected."""
        mae = MeanAbsoluteError()
        mse = MeanSquaredError()
        metrics = MetricSet([mae, mse])
        config = metrics.get_config()
        m_conf = [(mae.name, mae.get_config()), (mse.name, mse.get_config())]
        assert isinstance(config, dict)
        assert config == {"metrics": m_conf}

    def test_from_config(self) -> None:
        """Test that `MetricSet.from_config` works as expected."""
        mae = MeanAbsoluteError()
        mse = MeanSquaredError()
        m_conf = [(mae.name, mae.get_config()), (mse.name, mse.get_config())]
        config = {"metrics": m_conf}
        metrics = MetricSet.from_config(config)
        assert isinstance(metrics, MetricSet)
        assert len(metrics.metrics) == 2
        assert isinstance(metrics.metrics[0], MeanAbsoluteError)
        assert isinstance(metrics.metrics[1], MeanSquaredError)
        assert metrics.get_config() == config

    def test_from_specs_none(self) -> None:
        """Test that `MetricSet.from_specs(None)` works as expected."""
        metrics = MetricSet.from_specs(None)
        assert isinstance(metrics, MetricSet)
        assert not metrics.metrics

    def test_from_specs_list(self) -> None:
        """Test that `MetricSet.from_specs([...])` works as expected."""
        mae = MeanAbsoluteError()
        metrics = MetricSet.from_specs([mae])
        assert isinstance(metrics, MetricSet)
        assert metrics.metrics == [mae]

    def test_from_specs_instance(self) -> None:
        """Test that `MetricSet.from_specs(MetricSet)` works as expected."""
        metrics = MetricSet(["mae"])
        assert MetricSet.from_specs(metrics) is metrics

    def test_from_specs_error(self) -> None:
        """Test that `MetricSet.from_specs` raises the expected TypeError."""
        with pytest.raises(TypeError):
            MetricSet.from_specs("invalid-specs")  # type: ignore
