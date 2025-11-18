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

"""Unit tests for 'declearn.main.config.FLOptimConfig'."""

import dataclasses
import os
from unittest import mock

import pytest

from declearn.aggregator import Aggregator, AveragingAggregator, SumAggregator
from declearn.fairness.api import FairnessControllerServer
from declearn.fairness.fairgrad import FairgradControllerServer
from declearn.main.config import FLOptimConfig
from declearn.optimizer import Optimizer
from declearn.optimizer.modules import AdamModule

FIELDS = {field.name: field for field in dataclasses.fields(FLOptimConfig)}


class TestFLOptimConfig:
    """Unit tests for 'declearn.main.config.FLOptimConfig'."""

    # unit tests; pylint: disable=too-many-public-methods

    # Client-side optimizer.

    def test_parse_client_opt_float(self) -> None:
        """Test parsing 'client_opt' from a float input."""
        field = FIELDS["client_opt"]
        optim = FLOptimConfig.parse_client_opt(field, 0.1)
        assert isinstance(optim, Optimizer)
        assert optim.lrate == 0.1
        assert optim.w_decay == 0.0
        assert not optim.modules
        assert not optim.regularizers

    def test_parse_client_opt_dict(self) -> None:
        """Test parsing 'client_opt' from a dict input."""
        field = FIELDS["client_opt"]
        config = {"lrate": 0.1, "modules": ["adam"]}
        optim = FLOptimConfig.parse_client_opt(field, config)
        assert isinstance(optim, Optimizer)
        assert optim.lrate == 0.1
        assert optim.w_decay == 0.0
        assert len(optim.modules) == 1
        assert isinstance(optim.modules[0], AdamModule)
        assert not optim.regularizers

    def test_parse_client_opt_dict_error(self) -> None:
        """Test parsing 'client_opt' from an invalid dict input."""
        field = FIELDS["client_opt"]
        config = {"modules": ["adam"]}  # missing 'lrate'
        with pytest.raises(TypeError):
            FLOptimConfig.parse_client_opt(field, config)

    def test_parse_client_opt_optimizer(self) -> None:
        """Test parsing 'client_opt' from an Optimizer input."""
        field = FIELDS["client_opt"]
        optim = mock.create_autospec(Optimizer, instance=True)
        assert FLOptimConfig.parse_client_opt(field, optim) is optim

    def test_parse_client_opt_error(self) -> None:
        """Test parsing 'client_opt' from an invalid-type input."""
        field = FIELDS["client_opt"]
        with pytest.raises(TypeError):
            FLOptimConfig.parse_client_opt(field, mock.MagicMock())

    # Server-side optimizer.
    # pylint: disable=duplicate-code

    def test_parse_server_opt_none(self) -> None:
        """Test parsing 'server_opt' from None."""
        field = FIELDS["server_opt"]
        optim = FLOptimConfig.parse_server_opt(field, None)
        assert isinstance(optim, Optimizer)
        assert optim.lrate == 1.0
        assert optim.w_decay == 0.0
        assert not optim.modules
        assert not optim.regularizers

    def test_parse_server_opt_float(self) -> None:
        """Test parsing 'server_opt' from a float input."""
        field = FIELDS["server_opt"]
        optim = FLOptimConfig.parse_server_opt(field, 0.1)
        assert isinstance(optim, Optimizer)
        assert optim.lrate == 0.1
        assert optim.w_decay == 0.0
        assert not optim.modules
        assert not optim.regularizers

    def test_parse_server_opt_dict(self) -> None:
        """Test parsing 'server_opt' from a dict input."""
        field = FIELDS["server_opt"]
        config = {"lrate": 0.1, "modules": ["adam"]}
        optim = FLOptimConfig.parse_server_opt(field, config)
        assert isinstance(optim, Optimizer)
        assert optim.lrate == 0.1
        assert optim.w_decay == 0.0
        assert len(optim.modules) == 1
        assert isinstance(optim.modules[0], AdamModule)
        assert not optim.regularizers

    def test_parse_server_opt_dict_error(self) -> None:
        """Test parsing 'server_opt' from an invalid dict input."""
        field = FIELDS["server_opt"]
        config = {"modules": ["adam"]}  # missing 'lrate'
        with pytest.raises(TypeError):
            FLOptimConfig.parse_server_opt(field, config)

    def test_parse_server_opt_optimizer(self) -> None:
        """Test parsing 'server_opt' from an Optimizer input."""
        field = FIELDS["server_opt"]
        optim = mock.create_autospec(Optimizer, instance=True)
        assert FLOptimConfig.parse_server_opt(field, optim) is optim

    def test_parse_server_opt_error(self) -> None:
        """Test parsing 'server_opt' from an invalid-type input."""
        field = FIELDS["server_opt"]
        with pytest.raises(TypeError):
            FLOptimConfig.parse_server_opt(field, mock.MagicMock())

    # pylint: enable=duplicate-code
    # Aggregator.

    def test_parse_aggregator_none(self) -> None:
        """Test parsing 'aggregator' from None."""
        field = FIELDS["aggregator"]
        aggregator = FLOptimConfig.parse_aggregator(field, None)
        assert isinstance(aggregator, AveragingAggregator)

    def test_parse_aggregator_str(self) -> None:
        """Test parsing 'aggregator' from a string."""
        field = FIELDS["aggregator"]
        aggregator = FLOptimConfig.parse_aggregator(field, "sum")
        assert isinstance(aggregator, SumAggregator)

    def test_parse_aggregator_dict(self) -> None:
        """Test parsing 'aggregator' from a dict."""
        field = FIELDS["aggregator"]
        config = {"name": "averaging", "config": {"steps_weighted": False}}
        aggregator = FLOptimConfig.parse_aggregator(field, config)
        assert isinstance(aggregator, AveragingAggregator)
        assert not aggregator.steps_weighted

    def test_parse_aggregator_dict_error(self) -> None:
        """Test parsing 'aggregator' from an invalid dict."""
        field = FIELDS["aggregator"]
        config = {"name": "adam", "group": "OptiModule"}  # wrong target type
        with pytest.raises(TypeError):
            FLOptimConfig.parse_aggregator(field, config)

    def test_parse_aggregator_aggregator(self) -> None:
        """Test parsing 'aggregator' from an Aggregator."""
        field = FIELDS["aggregator"]
        aggregator = mock.create_autospec(Aggregator, instance=True)
        assert FLOptimConfig.parse_aggregator(field, aggregator) is aggregator

    def test_parse_aggregator_error(self) -> None:
        """Test parsing 'aggregator' from an invalid-type input."""
        field = FIELDS["aggregator"]
        with pytest.raises(TypeError):
            FLOptimConfig.parse_aggregator(field, mock.MagicMock())

    # Fairness.

    def test_parse_fairness_none(self) -> None:
        """Test parsing 'fairness' from None."""
        field = FIELDS["fairness"]
        fairness = FLOptimConfig.parse_fairness(field, None)
        assert fairness is None

    def test_parse_fairness_dict(self) -> None:
        """Test parsing 'fairness' from a dict."""
        field = FIELDS["fairness"]
        config = {
            "algorithm": "fairgrad",
            "f_type": "demographic_parity",
            "eta": 0.1,
            "eps": 0.0,
        }
        fairness = FLOptimConfig.parse_fairness(field, config)
        assert isinstance(fairness, FairgradControllerServer)
        assert fairness.f_type == "demographic_parity"
        assert fairness.weights_controller.eta == 0.1
        assert fairness.weights_controller.eps == 0.0

    def test_parse_fairness_dict_error(self) -> None:
        """Test parsing 'fairness' from an invalid dict."""
        field = FIELDS["fairness"]
        config = {"algorithm": "fairgrad"}  # missing f_type choice
        with pytest.raises(TypeError):
            FLOptimConfig.parse_fairness(field, config)

    def test_parse_fairness_controller(self) -> None:
        """Test parsing 'fairness' from a FairnessControllerServer."""
        field = FIELDS["fairness"]
        fairness = mock.create_autospec(
            FairnessControllerServer, instance=True
        )
        assert FLOptimConfig.parse_fairness(field, fairness) is fairness

    # Functional test.

    def test_from_toml(self, tmp_path: str) -> None:
        """Test parsing an arbitrary, complex TOML file."""
        # Set up an arbitrary TOML file parseabld into an FLOptimConfig.
        path = os.path.join(tmp_path, "config.toml")
        toml_config = """
        [optim]
        aggregator = "sum"
        client_opt = 0.001
        [optim.server_opt]
            lrate = 1.0
            modules = [["adam", {beta_1=0.8, beta_2=0.9}]]
        [optim.fairness]
            algorithm = "fairgrad"
            f_type = "equalized_odds"
            eta = 0.1
            eps = 0.0
        """
        with open(path, "w", encoding="utf-8") as file:
            file.write(toml_config)
        # Parse the TOML file and verify that outputs match expectations.
        optim = FLOptimConfig.from_toml(path, use_section="optim")
        assert isinstance(optim, FLOptimConfig)
        assert isinstance(optim.aggregator, SumAggregator)
        assert isinstance(optim.client_opt, Optimizer)
        assert optim.client_opt.lrate == 0.001
        assert not optim.client_opt.modules
        assert isinstance(optim.server_opt, Optimizer)
        assert optim.server_opt.lrate == 1.0
        assert len(optim.server_opt.modules) == 1
        assert isinstance(optim.server_opt.modules[0], AdamModule)
        assert optim.server_opt.modules[0].ewma_1.beta == 0.8
        assert optim.server_opt.modules[0].ewma_2.beta == 0.9
        assert isinstance(optim.fairness, FairgradControllerServer)
        assert optim.fairness.f_type == "equalized_odds"
        assert optim.fairness.weights_controller.eta == 0.1
        assert optim.fairness.weights_controller.eps == 0.0
