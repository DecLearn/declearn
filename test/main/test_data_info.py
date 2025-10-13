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

"""Unit tests for 'declearn.main.utils.aggregate_clients_data_info'."""

from unittest import mock

import pytest

from declearn.main.utils import AggregationError, aggregate_clients_data_info


class TestAggregateClientsDataInfo:
    """Unit tests for 'declearn.main.utils.aggregate_clients_data_info'."""

    def test_with_valid_inputs(self) -> None:
        """Test 'aggregate_clients_data_info' with valid inputs."""
        clients_data_info = {
            "client_a": {"n_samples": 10},
            "client_b": {"n_samples": 32},
        }
        results = aggregate_clients_data_info(clients_data_info, {"n_samples"})
        assert results == {"n_samples": 42}

    def test_with_missing_fields(self) -> None:
        """Test 'aggregate_clients_data_info' with some missing fields."""
        clients_data_info = {
            "client_a": {"n_samples": 10},
            "client_b": {"n_samples": 32},
        }
        with pytest.raises(AggregationError):
            aggregate_clients_data_info(
                clients_data_info,
                required_fields={"n_samples", "features_shape"},
            )

    def test_with_invalid_values(self) -> None:
        """Test 'aggregate_clients_data_info' with some invalid values."""
        clients_data_info = {
            "client_a": {"n_samples": 10},
            "client_b": {"n_samples": -1},
        }
        with pytest.raises(AggregationError):
            aggregate_clients_data_info(
                clients_data_info, required_fields={"n_samples"}
            )

    def test_with_incompatible_values(self) -> None:
        """Test 'aggregate_clients_data_info' with some incompatible values."""
        clients_data_info = {
            "client_a": {"features_shape": (100,)},
            "client_b": {"features_shape": (128,)},
        }
        with pytest.raises(AggregationError):
            aggregate_clients_data_info(
                clients_data_info, required_fields={"features_shape"}
            )

    def test_with_unexpected_keyerror(self) -> None:
        """Test 'aggregate_clients_data_info' with an unforeseen KeyError."""
        with mock.patch(
            "declearn.main.utils._data_info.aggregate_data_info"
        ) as patch_agg:
            patch_agg.side_effect = KeyError("Forced KeyError")
            with pytest.raises(AggregationError):
                aggregate_clients_data_info(
                    clients_data_info={"client_a": {}, "client_b": {}},
                    required_fields=set(),
                )

    def test_with_unexpected_exception(self) -> None:
        """Test 'aggregate_clients_data_info' with an unforeseen Exception."""
        with mock.patch(
            "declearn.main.utils._data_info.aggregate_data_info"
        ) as patch_agg:
            patch_agg.side_effect = Exception("Forced Exception")
            with pytest.raises(AggregationError):
                aggregate_clients_data_info(
                    clients_data_info={"client_a": {}, "client_b": {}},
                    required_fields=set(),
                )
