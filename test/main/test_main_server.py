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

"""Unit tests for 'FederatedServer'."""

import logging
import os
from unittest import mock

import pytest  # type: ignore

from declearn.aggregator import Aggregator
from declearn.communication import NetworkServerConfig
from declearn.communication.api import NetworkServer
from declearn.main import FederatedServer
from declearn.main.config import FLOptimConfig
from declearn.main.utils import Checkpointer
from declearn.metrics import MetricSet
from declearn.model.api import Model
from declearn.model.sklearn import SklearnSGDModel
from declearn.optimizer import Optimizer
from declearn.secagg.api import SecaggConfigServer
from declearn.utils import serialize_object


MOCK_MODEL = mock.create_autospec(Model, instance=True)
MOCK_NETWK = mock.create_autospec(NetworkServer, instance=True)
MOCK_NETWK.name = "server"
MOCK_OPTIM = FLOptimConfig(
    client_opt=mock.create_autospec(Optimizer, instance=True),
    server_opt=mock.create_autospec(Optimizer, instance=True),
    aggregator=mock.create_autospec(Aggregator, instance=True),
)


class TestFederatedServerInit:  # pylint: disable=too-many-public-methods
    """Unit tests for 'FederatedServer.__init__'."""

    # Tests for the 'model' argument.

    def test_model_instance(self) -> None:
        """Test specifying 'model' as a 'Model' instance."""
        model = SklearnSGDModel.from_parameters(kind="regressor")
        server = FederatedServer(
            model=model, netwk=MOCK_NETWK, optim=MOCK_OPTIM
        )
        assert server.model is model

    def test_model_serialized(self) -> None:
        """Test specifying 'model' as a serialized 'Model'."""
        model = SklearnSGDModel.from_parameters(kind="regressor")
        serialized = dict(serialize_object(model).to_dict())
        server = FederatedServer(
            model=serialized, netwk=MOCK_NETWK, optim=MOCK_OPTIM
        )
        assert isinstance(server.model, SklearnSGDModel)
        assert server.model.get_config() == model.get_config()

    def test_model_json_path(self, tmp_path: str) -> None:
        """Test specifying 'model' as a serialized 'Model' file path."""
        model = SklearnSGDModel.from_parameters(kind="regressor")
        path = os.path.join(tmp_path, "model.json")
        serialize_object(model).to_json(path)
        server = FederatedServer(
            model=path, netwk=MOCK_NETWK, optim=MOCK_OPTIM
        )
        assert isinstance(server.model, SklearnSGDModel)
        assert server.model.get_config() == model.get_config()

    def test_model_invalid(self) -> None:
        """Test specifying 'model' with an invalid type."""
        with pytest.raises(TypeError):
            FederatedServer(
                model=mock.MagicMock(), netwk=MOCK_NETWK, optim=MOCK_OPTIM
            )

    # Tests for the 'netwk' argument.

    def test_netwk_instance(self) -> None:
        """Test specifying 'netwk' as a 'NetworkServer' instance."""
        netwk = mock.create_autospec(NetworkServer, instance=True)
        netwk.name = "server"
        server = FederatedServer(
            model=MOCK_MODEL, netwk=netwk, optim=MOCK_OPTIM
        )
        assert server.netwk is netwk

    def test_netwk_config(self) -> None:
        """Test specifying 'netwk' as a 'NetworkServerConfig' instance."""
        netwk = mock.create_autospec(NetworkServerConfig, instance=True)
        server = FederatedServer(
            model=MOCK_MODEL, netwk=netwk, optim=MOCK_OPTIM
        )
        netwk.build_server.assert_called_once()
        assert server.netwk is netwk.build_server.return_value

    def test_netwk_config_dict(self) -> None:
        """Test specifying 'netwk' as a properly-parsable dict."""
        netwk = {"protocol": "mock", "host": "host", "port": 8000}
        with mock.patch.object(NetworkServerConfig, "build_server") as patched:
            server = FederatedServer(
                model=MOCK_MODEL, netwk=netwk, optim=MOCK_OPTIM
            )
        patched.assert_called_once()
        assert server.netwk is patched.return_value

    def test_netwk_config_file(self) -> None:
        """Test specifying 'netwk' as a path to a TOML file."""
        netwk = "stub_path_to_netwk_config.toml"
        with mock.patch.object(NetworkServerConfig, "from_toml") as patched:
            server = FederatedServer(
                model=MOCK_MODEL, netwk=netwk, optim=MOCK_OPTIM
            )
        patched.assert_called_once_with(netwk)
        patched.return_value.build_server.assert_called_once()
        assert server.netwk is patched.return_value.build_server.return_value

    def test_netwk_config_invalid(self) -> None:
        """Test specifying 'netwk' as an invalid type."""
        with pytest.raises(TypeError):
            FederatedServer(
                model=MOCK_MODEL, netwk=mock.MagicMock(), optim=MOCK_OPTIM
            )

    # Tests for the 'optim' argument.

    def test_optim_instance(self) -> None:
        """Test specifying 'optim' as a 'FLOptimConfig' instance."""
        optim = FLOptimConfig(
            client_opt=mock.create_autospec(Optimizer, instance=True),
            server_opt=mock.create_autospec(Optimizer, instance=True),
            aggregator=mock.create_autospec(Aggregator, instance=True),
        )
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=optim
        )
        assert server.c_opt is optim.client_opt
        assert server.optim is optim.server_opt
        assert server.aggrg is optim.aggregator

    def test_optim_dict(self) -> None:
        """Test specifying 'optim' as a config dict."""
        optim = {
            "client_opt": mock.create_autospec(Optimizer, instance=True),
            "server_opt": mock.create_autospec(Optimizer, instance=True),
            "aggregator": mock.create_autospec(Aggregator, instance=True),
        }
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=optim
        )
        assert server.c_opt is optim["client_opt"]
        assert server.optim is optim["server_opt"]
        assert server.aggrg is optim["aggregator"]

    def test_optim_toml(self, tmp_path: str) -> None:
        """Test specifying 'optim' as a TOML file path."""
        # Set up a valid FLOptimConfig TOML file.
        toml_file = """
        [client_opt]
        lrate = 0.01
        modules = ["adam"]

        [server_opt]
        lrate = 1.0

        [aggregator]
        name = "averaging"
        steps_weighted = false
        """
        path = os.path.join(tmp_path, "optim.toml")
        with open(path, "w", encoding="utf-8") as file:
            file.write(toml_file)
        # Try instantiating from its path.
        config = FLOptimConfig.from_toml(path)
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=path
        )
        assert server.c_opt.get_config() == config.client_opt.get_config()
        assert server.optim.get_config() == config.server_opt.get_config()
        assert server.aggrg.get_config() == config.aggregator.get_config()

    def test_optim_invalid(self) -> None:
        """Test specifying 'optim' with an invalid type."""
        with pytest.raises(TypeError):
            FederatedServer(
                model=MOCK_MODEL, netwk=MOCK_NETWK, optim=mock.MagicMock()
            )

    # Tests for the 'metrics' argument.

    def test_metrics_instance(self) -> None:
        """Test specifying 'metrics' as a MetricSet instance."""
        metrics = mock.create_autospec(MetricSet, instance=True)
        server = FederatedServer(
            MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, metrics=metrics
        )
        assert server.metrics is metrics

    def test_metrics_specs(self) -> None:
        """Test specifying 'metrics' as a list of specs.

        Note: 'MetricSet.from_specs' has its own unit tests.
        """
        metrics = ["binary-classif", "binary-roc"]
        with mock.patch.object(MetricSet, "from_specs") as patched:
            server = FederatedServer(
                # fmt: off
                MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM,
                metrics=metrics  # type: ignore[arg-type]
            )
        patched.assert_called_once_with(metrics)
        assert server.metrics is patched.return_value

    def test_metrics_none(self) -> None:
        """Test specifying 'metrics' as None."""
        server = FederatedServer(
            MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, metrics=None
        )
        assert isinstance(server.metrics, MetricSet)
        assert not server.metrics.metrics

    def test_metrics_invalid(self) -> None:
        """Test specifying 'metrics' as a MetricSet instance."""
        with pytest.raises(TypeError):
            FederatedServer(
                MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, metrics=mock.MagicMock()
            )

    # Tests for the 'secagg' argument.

    def test_secagg_instance(self) -> None:
        """Test specifying 'secagg' as a SecaggConfigServer instance."""
        secagg = mock.create_autospec(SecaggConfigServer, instance=True)
        server = FederatedServer(
            MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, secagg=secagg
        )
        assert server.secagg is secagg

    def test_secagg_none(self) -> None:
        """Test specifying 'secagg' as None."""
        server = FederatedServer(
            MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, secagg=None
        )
        assert server.secagg is None

    def test_secagg_dict(self) -> None:
        """Test specifying 'secagg' as a config dict."""
        secagg = {"secagg_type": "mock"}
        with mock.patch(
            "declearn.main._server.parse_secagg_config_server"
        ) as patched:
            server = FederatedServer(
                MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, secagg=secagg
            )
        patched.assert_called_once_with(**secagg)
        assert server.secagg is patched.return_value

    def test_secagg_invalid(self) -> None:
        """Test specifying 'secagg' as an invalid type."""
        with pytest.raises(TypeError):
            FederatedServer(
                MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, secagg=mock.MagicMock()
            )

    # Tests for the 'checkpoint' argument.

    def test_checkpoint_instance(self) -> None:
        """Test specifying 'checkpoint' as a Checkpointer instance."""
        checkpointer = mock.create_autospec(Checkpointer, instance=True)
        server = FederatedServer(
            MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, checkpoint=checkpointer
        )
        assert server.ckptr is checkpointer

    def test_checkpoint_none(self) -> None:
        """Test specifying 'checkpoint' as None."""
        server = FederatedServer(
            MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, checkpoint=None
        )
        assert server.ckptr is None

    def test_checkpoint_specs(self) -> None:
        """Test specifying 'checkpoint' as some specs.

        Note: 'Checkpointer.from_specs' has its own unit tests for subcases.
        """
        specs = {"folder": "mock_folder", "max_history": 1}
        with mock.patch.object(Checkpointer, "from_specs") as patched:
            server = FederatedServer(
                MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, checkpoint=specs
            )
        patched.assert_called_once_with(specs)
        assert server.ckptr is patched.return_value

    # Tests for the 'logger' argument.

    def test_logger_instance(self) -> None:
        """Test specifying 'logger' as a Logger instance."""
        logger = logging.Logger("mock-server-logger")
        server = FederatedServer(
            MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, logger=logger
        )
        assert server.logger is logger

    def test_logger_str(self) -> None:
        """Test specifying 'logger' as a logger name."""
        logger = "mock-client-logger"
        server = FederatedServer(
            MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, logger=logger
        )
        assert isinstance(server.logger, logging.Logger)
        assert server.logger.name == logger

    def test_logger_none(self) -> None:
        """Test specifying 'logger' as None."""
        server = FederatedServer(
            MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, logger=None
        )
        assert isinstance(server.logger, logging.Logger)

    def test_logger_invalid(self) -> None:
        """Test specifying 'logger' with a wrong type."""
        with pytest.raises(TypeError):
            FederatedServer(
                MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, logger=mock.MagicMock()
            )
