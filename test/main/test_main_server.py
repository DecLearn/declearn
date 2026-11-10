# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

import math
import os
from typing import Dict, List, Optional, Type
from unittest import mock

import pytest  # type: ignore

from declearn.aggregator import Aggregator, ModelUpdates
from declearn.communication import NetworkServerConfig
from declearn.communication.api import NetworkServer
from declearn.fairness.api import FairnessControllerServer
from declearn.main import FederatedServer
from declearn.main.config import (
    EvaluateConfig,
    FairnessConfig,
    FLOptimConfig,
    FLRunConfig,
    RegisterConfig,
    TrainingConfig,
)
from declearn.main.utils import Checkpointer
from declearn.messaging import (
    EvaluationReply,
    InitReply,
    InitRequest,
    Message,
    MetadataQuery,
    MetadataReply,
    PrivacyReply,
    PrivacyRequest,
    SerializedMessage,
    StopTraining,
    TrainReply,
)
from declearn.messaging._base import (
    MultiModelEvaluationReply,
    MultiModelEvaluationRequest,
    MultiModelTrainReply,
    MultiModelTrainRequest,
)
from declearn.messaging._fairness import MultiModelFairnessQuery
from declearn.metrics import MetricSet
from declearn.model.api import Model
from declearn.model.sklearn import SklearnSGDModel
from declearn.optimizer import Optimizer
from declearn.secagg.api import Decrypter, SecaggConfigServer
from declearn.secagg.messaging import (
    MultiModelSecaggEvaluationReply,
    MultiModelSecaggTrainReply,
    SecaggEvaluationReply,
    SecaggTrainReply,
)
from declearn.utils.serialize import json_dump

MOCK_MODEL = mock.create_autospec(Model, instance=True)
MOCK_NETWK = mock.create_autospec(NetworkServer, instance=True)
MOCK_NETWK.name = "server"
MOCK_NETWK.client_names = {"client1", "client2"}
MOCK_OPTIM = FLOptimConfig(
    client_opt=mock.create_autospec(Optimizer, instance=True),
    server_opt=mock.create_autospec(Optimizer, instance=True),
    aggregator=mock.create_autospec(Aggregator, instance=True),
)
# Necessary for the mocking the deep-copied aggregator.
MOCK_OPTIM.aggregator.__deepcopy__ = lambda memo: mock.create_autospec(
    Aggregator, instance=True
)


class TestFLRunConfig:
    """Unit tests for FLRunConfig"""

    def test_config_dict_list(self):
        """Tests wether the creation of FLRunConfig is
        done correctly given a list of training and evaluate"""
        config = FLRunConfig.from_params(
            rounds=1,
            register={"min_clients": 1},
            training=[
                {"batch_size": 30, "n_epoch": 1},
                {"batch_size": 30, "n_epoch": 1},
            ],
            evaluate=[
                {
                    "batch_size": 30,
                },
                {
                    "batch_size": 30,
                },
            ],
        )
        print("\n", config.evaluate)
        assert isinstance(config.training, List)
        assert isinstance(config.training[0], TrainingConfig)
        assert len(config.training) == 2
        assert isinstance(config.evaluate, List)
        assert isinstance(config.evaluate[0], EvaluateConfig)
        assert len(config.evaluate) == 2

    def test_config_list_training_evaluate_mismatch(self):
        """Tests wether internal __post_init__ raises an error
        in case of length mismatch"""
        with pytest.raises(ValueError):
            FLRunConfig.from_params(
                rounds=1,
                register={"min_clients": 1},
                training=[
                    {"batch_size": 30, "n_epoch": 1},
                    {"batch_size": 30, "n_epoch": 1},
                ],
                evaluate=[
                    {
                        "batch_size": 30,
                    },
                ],
            )

    def test_config_list_evaluate_frequency_mismatch(
        self,
    ):
        """Tests wether internal __post_init__ raises an error
        in case of frequency mismatch"""
        with pytest.raises(ValueError):
            FLRunConfig.from_params(
                rounds=1,
                register={"min_clients": 1},
                training=[
                    {"batch_size": 30, "n_epoch": 1},
                    {"batch_size": 30, "n_epoch": 1},
                ],
                evaluate=[
                    {"batch_size": 30, "frequency": 1},
                    {"batch_size": 30, "frequency": 2},
                ],
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
        assert server.model == [model]

    def test_model_instances(
        self,
    ):
        """Test specifying a list of 'Model' instances"""
        models = [
            SklearnSGDModel.from_parameters(kind="regressor"),
            SklearnSGDModel.from_parameters(kind="classifier"),
        ]
        server = FederatedServer(
            model=models, netwk=MOCK_NETWK, optim=MOCK_OPTIM
        )
        assert server.model == models
        assert len(server.model) == 2

    def test_model_serialized(self) -> None:
        """Test specifying 'model' as a serialized 'Model'."""
        model = SklearnSGDModel.from_parameters(kind="regressor")
        config = model.get_config(allow_bin=False)
        server = FederatedServer(
            model=[config], netwk=MOCK_NETWK, optim=MOCK_OPTIM
        )
        assert isinstance(server.model, List)
        assert isinstance(server.model[0], SklearnSGDModel)
        assert server.model[0].get_config() == model.get_config()

    def test_model_json_path(self, tmp_path: str) -> None:
        """Test specifying 'model' as a serialized 'Model' file path."""
        model = SklearnSGDModel.from_parameters(kind="regressor")
        config = model.get_config(allow_bin=False)
        path = os.path.join(tmp_path, "model.json")
        json_dump(config, path)
        server = FederatedServer(
            model=[path], netwk=MOCK_NETWK, optim=MOCK_OPTIM
        )
        assert isinstance(server.model[0], SklearnSGDModel)
        assert server.model[0].get_config() == model.get_config()

    def test_model_invalid(self) -> None:
        """Test specifying 'model' with an invalid type."""
        with pytest.raises(TypeError):
            FederatedServer(
                model=[mock.MagicMock()],
                netwk=MOCK_NETWK,
                optim=MOCK_OPTIM,
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
                model=MOCK_MODEL,
                netwk=mock.MagicMock(),
                optim=MOCK_OPTIM,
            )

    # Tests for the 'optim' argument.

    def test_optim_instance_single_model(self) -> None:
        """Test specifying 'optim' as a 'FLOptimConfig' instance."""
        optim = FLOptimConfig(
            client_opt=mock.create_autospec(Optimizer, instance=True),
            server_opt=mock.create_autospec(Optimizer, instance=True),
            aggregator=mock.create_autospec(Aggregator, instance=True),
            fairness=mock.create_autospec(
                FairnessControllerServer, instance=True
            ),
        )
        optim.client_opt.from_config.return_value = mock.create_autospec(
            Optimizer, instance=True
        )
        optim.server_opt.from_config.return_value = mock.create_autospec(
            Optimizer, instance=True
        )
        optim.aggregator.from_config.return_value = mock.create_autospec(
            Aggregator, instance=True
        )
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=optim
        )

        assert isinstance(server.c_opt[0], Optimizer)
        assert isinstance(server.optim[0], Optimizer)
        assert isinstance(server.aggrg[0], Aggregator)
        assert server.fairness == optim.fairness
        # check if the instances are fresh copies:
        assert server.c_opt[0] is not optim.client_opt
        assert server.optim[0] is not optim.server_opt
        assert server.aggrg[0] is not optim.aggregator

    def test_optim_list_multiple_models(self):
        """Test specyfing 'client_optim' and 'server_optim' as lists
        for the case of multiple models"""
        optim = FLOptimConfig(
            client_opt=[
                mock.create_autospec(Optimizer, instance=True),
                mock.create_autospec(Optimizer, instance=True),
            ],
            server_opt=[
                mock.create_autospec(Optimizer, instance=True),
                mock.create_autospec(Optimizer, instance=True),
            ],
            aggregator=mock.create_autospec(Aggregator, instance=True),
            fairness=mock.create_autospec(
                FairnessControllerServer, instance=True
            ),
        )
        optim.client_opt[0].from_config.return_value = mock.create_autospec(
            Optimizer, instance=True
        )
        optim.server_opt[0].from_config.return_value = mock.create_autospec(
            Optimizer, instance=True
        )
        optim.aggregator.from_config.return_value = mock.create_autospec(
            Aggregator, instance=True
        )
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=optim
        )
        assert server.c_opt == optim.client_opt
        assert server.optim == optim.server_opt
        assert isinstance(server.aggrg[0], Aggregator)
        assert server.aggrg[0] is not optim.aggregator
        assert server.fairness == optim.fairness

    def test_optim_dict(self) -> None:
        """Test specifying 'optim' as a config dict."""
        optim = {
            "client_opt": mock.create_autospec(Optimizer, instance=True),
            "server_opt": mock.create_autospec(Optimizer, instance=True),
            "aggregator": mock.create_autospec(Aggregator, instance=True),
            "fairness": mock.create_autospec(
                FairnessControllerServer, instance=True
            ),
        }
        optim["client_opt"].from_config.return_value = mock.create_autospec(
            Optimizer, instance=True
        )
        optim["server_opt"].from_config.return_value = mock.create_autospec(
            Optimizer, instance=True
        )
        optim["aggregator"].from_config.return_value = mock.create_autospec(
            Aggregator, instance=True
        )
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=optim
        )
        assert server.c_opt[0] is not optim["client_opt"]
        assert server.optim[0] is not optim["server_opt"]
        assert server.aggrg[0] is not optim["aggregator"]
        assert server.fairness is optim["fairness"]
        assert isinstance(server.c_opt[0], Optimizer)
        assert isinstance(server.optim[0], Optimizer)
        assert isinstance(server.aggrg[0], Aggregator)

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
        assert server.c_opt[0].get_config() == config.client_opt.get_config()
        assert server.optim[0].get_config() == config.server_opt.get_config()
        assert server.aggrg[0].get_config() == config.aggregator.get_config()
        # verify that for every model, the parsed_optim.fairness is appended
        assert server.fairness is None

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
        assert server.metrics == metrics

    def test_metrics_specs(self) -> None:
        """Test specifying 'metrics' as a list of specs.

        Note: 'MetricSet.from_specs' has its own unit tests.
        """
        metrics = ["binary-classif", "binary-roc"]
        with mock.patch.object(MetricSet, "from_specs") as patched:
            # fmt: off
            server = FederatedServer(
                MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM,
                metrics=metrics  # type: ignore[arg-type]
            )
            # fmt: on
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
                MOCK_MODEL,
                MOCK_NETWK,
                MOCK_OPTIM,
                metrics=mock.MagicMock(),
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
            "declearn.main.utils._server.ServerUtils.parse_secagg"
        ) as patched:
            server = FederatedServer(
                MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, secagg=secagg
            )
        patched.assert_called_once_with(secagg)
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

        Note: 'Checkpointer.from_specs' has its own unit tests for sub-cases.
        """
        specs = {"folder": "mock_folder", "max_history": 1}
        with mock.patch.object(Checkpointer, "from_specs") as patched:
            server = FederatedServer(
                MOCK_MODEL, MOCK_NETWK, MOCK_OPTIM, checkpoint=specs
            )
        patched.assert_called_once_with(specs)
        assert server.ckptr is patched.return_value


class TestFederatedServerRoutines:
    """Unit tests for 'FederatedServer' main unitary routines."""

    @staticmethod
    async def setup_test_server_single_model_instance(
        use_secagg: bool = False,
        use_fairness: bool = False,
    ) -> FederatedServer:
        """Set up a FederatedServer wrapping mock controllers."""
        netwk = mock.create_autospec(NetworkServer, instance=True)
        netwk.name = "server"
        netwk.client_names = {"client_a", "client_b"}

        model = mock.create_autospec(Model, instance=True)

        optim = FLOptimConfig(
            client_opt=mock.create_autospec(Optimizer, instance=True),
            server_opt=mock.create_autospec(Optimizer, instance=True),
            aggregator=mock.create_autospec(Aggregator, instance=True),
            fairness=(
                mock.create_autospec(FairnessControllerServer, instance=True)
                if use_fairness
                else None
            ),
        )
        optim.client_opt.from_config.return_value = mock.create_autospec(
            Optimizer, instance=True
        )
        optim.server_opt.from_config.return_value = mock.create_autospec(
            Optimizer, instance=True
        )
        optim.aggregator.from_config.return_value = mock.create_autospec(
            Aggregator, instance=True
        )
        secagg: Optional[SecaggConfigServer] = None
        if use_secagg:
            secagg = mock.create_autospec(SecaggConfigServer, instance=True)
            secagg.secagg_type = "mock_secagg"  # type: ignore

        return FederatedServer(
            model=model,
            netwk=netwk,
            optim=optim,
            metrics=mock.create_autospec(MetricSet, instance=True),
            secagg=secagg,
            checkpoint=mock.create_autospec(Checkpointer, instance=True),
        )

    @staticmethod
    async def setup_test_server_multiple_models(
        n_model: int = 2,
        use_secagg: bool = False,
        use_fairness: bool = False,
    ) -> FederatedServer:
        """Set up a FederatedServer wrapping mock controllers for the case
        of training multiple models with multiple configurations"""
        netwk = mock.create_autospec(NetworkServer, instance=True)
        netwk.name = "server"
        netwk.client_names = {"client_a", "client_b"}

        models = [
            mock.create_autospec(Model, instance=True) for _ in range(n_model)
        ]
        optim = FLOptimConfig(
            client_opt=[
                mock.create_autospec(Optimizer, instance=True),
                mock.create_autospec(Optimizer, instance=True),
            ],
            server_opt=[
                mock.create_autospec(Optimizer, instance=True),
                mock.create_autospec(Optimizer, instance=True),
            ],
            aggregator=mock.create_autospec(Aggregator, instance=True),
            fairness=(
                mock.create_autospec(FairnessControllerServer, instance=True)
                if use_fairness
                else None
            ),
        )
        optim.client_opt[0].from_config.return_value = mock.create_autospec(
            Optimizer, instance=True
        )
        optim.server_opt[0].from_config.return_value = mock.create_autospec(
            Optimizer, instance=True
        )
        optim.aggregator.from_config.return_value = mock.create_autospec(
            Aggregator, instance=True
        )

        secagg: Optional[SecaggConfigServer] = None
        if use_secagg:
            secagg = mock.create_autospec(SecaggConfigServer, instance=True)
            secagg.secagg_type = "mock_secagg"  # type: ignore

        return FederatedServer(
            model=models,
            netwk=netwk,
            optim=optim,
            metrics=mock.create_autospec(MetricSet, instance=True),
            secagg=secagg,
            checkpoint=mock.create_autospec(Checkpointer, instance=True),
        )

    @staticmethod
    def setup_mock_serialized_message(
        msg_cls: Type[Message],
        wrapped: Optional[Message] = None,
    ) -> mock.NonCallableMagicMock:
        """Set up a mock SerializedMessage with given wrapped message type."""
        message = mock.create_autospec(SerializedMessage, instance=True)
        message.message_cls = msg_cls
        if wrapped is None:
            wrapped = mock.create_autospec(msg_cls, instance=True)
        message.deserialize.return_value = wrapped
        return message

    @pytest.mark.parametrize(
        "metadata", [False, True], ids=["nometa", "metadata"]
    )
    @pytest.mark.parametrize("privacy", [False, True], ids=["nodp", "dpsgd"])
    @pytest.mark.parametrize(
        "fairness", [False, True], ids=["unfair", "fairness"]
    )
    @pytest.mark.parametrize("secagg", [False, True], ids=["clrtxt", "secagg"])
    @pytest.mark.asyncio
    async def test_initialization(
        self,
        secagg: bool,
        fairness: bool,
        privacy: bool,
        metadata: bool,
    ) -> None:
        """Test initialization with one or multiple model."""
        server = await self.setup_test_server_single_model_instance(
            use_secagg=secagg,
            use_fairness=fairness,
        )

        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert server._n_models == 1

        # Configure each model's metadata requirements.
        for model in server.model:
            model.required_data_info = {"n_samples"} if metadata else {}
        aggrg = server.aggrg[0]
        # Run initialization.
        config = FLRunConfig.from_params(
            rounds=10,
            register=RegisterConfig(0, 2, 120),
            training={"batch_size": 8},
            evaluate={"batch_size": 8},
            privacy=(
                {"budget": (1e-3, 0.0), "sclip_norm": 1.0} if privacy else None
            ),
        )

        server.netwk.wait_for_messages.side_effect = self._setup_init_replies(
            metadata=metadata,
            privacy=privacy,
            n_model=1,
        )
        await server.initialization(config)
        # Verify client registration.
        server.netwk.wait_for_clients.assert_awaited_once_with(0, 2, 120)
        # Verify message exchanges.
        assert server.netwk.broadcast_message.await_count == (
            1 + metadata + privacy
        )

        queries = server.netwk.broadcast_message.await_args_list.copy()
        # Verify metadata collection and model initialization.
        if metadata:
            query = queries.pop(0)[0][0]
            assert isinstance(query, MetadataQuery)
            assert len(query.fields) == 1
            assert query.fields[f"model_{0}"] == {"n_samples"}

            server.model[0].initialize.assert_called_once_with(
                {"n_samples": 200}
            )

        # Verify InitRequest.
        query = queries.pop(0)[0][0]
        assert isinstance(query, InitRequest)
        assert query.dpsgd is privacy

        if secagg:
            assert query.secagg is not None
        else:
            assert query.secagg is None
        assert query.fairness is fairness

        # Verify DP-SGD setup.
        if privacy:
            query = queries.pop(0)[0][0]
            assert isinstance(query, PrivacyRequest)
            assert query.budget == (1e-3, 0.0)
            assert query.sclip_norm == 1.0
            assert query.rounds == 10

        # Verify SecAgg setup.
        decrypter: Optional[Decrypter] = None
        if secagg:
            assert isinstance(server.secagg, mock.NonCallableMagicMock)
            if fairness:
                server.secagg.setup_decrypter.assert_awaited_once()
                decrypter = server.secagg.setup_decrypter.return_value
            else:
                server.secagg.setup_decrypter.assert_not_called()
        # Verify fairness setup across all the model
        # Verify that fairness setup occurred when expected.
        if fairness:
            assert isinstance(server.fairness, mock.NonCallableMagicMock)
            server.fairness.setup_fairness.assert_awaited_once_with(
                netwk=server.netwk, aggregator=aggrg, secagg=decrypter
            )
            assert (
                server.aggrg[0] is server.fairness.setup_fairness.return_value
            )

    def _setup_init_replies(
        self,
        metadata: bool,
        privacy: bool,
        n_model: int = 1,
    ) -> List[Dict[str, Dict[str, mock.NonCallableMagicMock]]]:
        """Prepare mocked client replies for initialization."""
        clients = ("client_a", "client_b")
        messages: List[Dict[str, Dict[str, mock.NonCallableMagicMock]]] = []

        if metadata:
            data = {
                f"model_{idx}": {"n_samples": 100} for idx in range(n_model)
            }

            msg = MetadataReply(data_info=data)
            messages.append(
                {
                    client: self.setup_mock_serialized_message(
                        MetadataReply, msg
                    )
                    for client in clients
                }
            )

        messages.append(
            {
                client: self.setup_mock_serialized_message(InitReply)
                for client in clients
            }
        )

        if privacy:
            messages.append(
                {
                    client: self.setup_mock_serialized_message(PrivacyReply)
                    for client in clients
                }
            )

        return messages

    @pytest.mark.parametrize("secagg", [False, True], ids=["clrtxt", "secagg"])
    @pytest.mark.asyncio
    async def test_training_round(
        self,
        secagg: bool,
    ) -> None:
        """Test that the 'training_round' routine triggers expected calls."""
        # Set up a server with mocked attributes.
        server = await self.setup_test_server_single_model_instance(
            use_secagg=secagg
        )

        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert isinstance(server.model, List)
        assert isinstance(server.model[0], mock.NonCallableMagicMock)
        assert isinstance(server.optim, List)
        assert isinstance(server.optim[0], mock.NonCallableMagicMock)
        assert isinstance(server.aggrg, List)
        assert isinstance(server.aggrg[0], mock.NonCallableMagicMock)
        # Mock-run a training routine.
        reply_cls: Type[Message] = (
            MultiModelSecaggTrainReply if secagg else MultiModelTrainReply  # type: ignore
        )
        updates = mock.create_autospec(ModelUpdates, instance=True)
        updates = mock.create_autospec(ModelUpdates, instance=True)
        reply_msg = (
            MultiModelTrainReply(
                replies=[
                    TrainReply(
                        n_epoch=1,
                        n_steps=10,
                        t_spent=0.0,
                        updates=updates,
                        aux_var={},
                    )
                ]
            )
            if not secagg
            else MultiModelSecaggTrainReply(
                replies=[
                    SecaggTrainReply(
                        n_steps=10,
                        t_spent=0.0,
                        updates=updates,
                        aux_var={},
                    )
                ]
            )
        )
        server.netwk.wait_for_messages.return_value = {
            "client_a": self.setup_mock_serialized_message(
                reply_cls, reply_msg
            ),
            "client_b": self.setup_mock_serialized_message(
                reply_cls, reply_msg
            ),
        }
        with mock.patch(
            "declearn.secagg.messaging.aggregate_secagg_messages",
            return_value=(reply_msg.replies[0] if secagg else reply_msg),
        ) as patch_aggregate_secagg_messages:
            await server.training_round(
                round_i=1,
                train_cfgs=[
                    TrainingConfig(batch_size=8) for _ in server.model
                ],
                clients=server.netwk.client_names,
            )
        # Verify that expected actions occurred.
        # (a) optional secagg setup
        if secagg:
            assert isinstance(server.secagg, mock.NonCallableMagicMock)
            server.secagg.setup_decrypter.assert_awaited_once()
        # (b) training request emission, including model weights
        server.netwk.send_messages.assert_awaited_once()
        queries = server.netwk.send_messages.await_args[0][0]
        assert isinstance(queries, dict)
        assert queries.keys() == server.netwk.client_names
        for query in queries.values():
            for reply in query.requests:
                assert isinstance(query, MultiModelTrainRequest)
                assert (
                    reply.weights is server.model[0].get_weights.return_value
                )
                assert (
                    reply.aux_var
                    is server.optim[0].collect_aux_var.return_value
                )
        # (c) training reply reception
        server.netwk.wait_for_messages.assert_awaited_once()
        if secagg:
            patch_aggregate_secagg_messages.assert_called_once()
        else:
            patch_aggregate_secagg_messages.assert_not_called()
        # (d) updates aggregation and global model weights update
        server.optim[0].process_aux_var.assert_called_once()
        server.aggrg[0].finalize_updates.assert_called_once()
        server.optim[0].apply_gradients.assert_called_once()

    @pytest.mark.parametrize("secagg", [False, True], ids=["clrtxt", "secagg"])
    @pytest.mark.asyncio
    async def test_evaluation_round(
        self,
        secagg: bool,
    ) -> None:
        """Test that the 'evaluation_round' routine triggers expected calls."""
        # Set up a server with mocked attributes.
        server = await self.setup_test_server_single_model_instance(
            use_secagg=secagg
        )
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert isinstance(server.model, List)
        assert isinstance(server.model[0], mock.NonCallableMagicMock)
        assert isinstance(server.metrics, MetricSet)
        assert isinstance(server.ckptr, mock.NonCallableMagicMock)
        # Mock-run an evaluation routine.
        reply_cls: Type[Message] = (
            MultiModelSecaggEvaluationReply  # type: ignore
            if secagg
            else MultiModelEvaluationReply
        )
        reply_msg = (
            MultiModelEvaluationReply(
                replies=[
                    EvaluationReply(
                        loss=0.42, n_steps=10, t_spent=0.0, metrics={}
                    )
                ]
            )
            if not secagg
            else MultiModelSecaggEvaluationReply(
                replies=[
                    SecaggEvaluationReply(
                        loss=0.42, n_steps=10, t_spent=0.0, metrics={}
                    )
                ]
            )
        )

        server.netwk.wait_for_messages.return_value = {
            "client_a": self.setup_mock_serialized_message(
                reply_cls, reply_msg
            ),
            "client_b": self.setup_mock_serialized_message(
                reply_cls, reply_msg
            ),
        }
        with mock.patch(
            "declearn.secagg.messaging.aggregate_secagg_messages",
            return_value=(reply_msg.replies[0] if secagg else reply_msg),
        ) as patch_aggregate_secagg_messages:
            await server.evaluation_round(
                round_i=1,
                valid_cfg=[EvaluateConfig(batch_size=8)],
                clients=server.netwk.client_names,
            )
        # Verify that expected actions occured.
        # (a) optional secagg setup
        if secagg:
            assert isinstance(server.secagg, mock.NonCallableMagicMock)
            server.secagg.setup_decrypter.assert_awaited_once()
        # (b) evaluation request emission, including model weights
        server.netwk.send_messages.assert_awaited_once()
        queries = server.netwk.send_messages.await_args[0][0]
        assert isinstance(queries, dict)
        assert queries.keys() == server.netwk.client_names
        for query in queries.values():
            assert isinstance(query, MultiModelEvaluationRequest)

            for reply in query.requests:
                assert (
                    reply.weights is server.model[0].get_weights.return_value
                )
        # (c) evaluation reply reception
        server.netwk.wait_for_messages.assert_awaited_once()
        if secagg:
            patch_aggregate_secagg_messages.assert_called_once()
        else:
            patch_aggregate_secagg_messages.assert_not_called()
        # (d) metrics aggregation
        server.metrics.reset.assert_called_once()
        server.metrics.set_states.assert_called_once()
        server.metrics.get_result.assert_called_once()
        # (e) checkpointing
        server.ckptr.checkpoint.assert_called_once_with(
            model=server.model[0],
            optimizer=server.optim[0],
            metrics=server.metrics.get_result.return_value,
            idx=None,
        )

    @pytest.mark.asyncio
    async def test_evaluation_round_skip(
        self,
    ) -> None:
        """Test that 'evaluation_round' skips rounds when configured."""
        # Set up a server with mocked attributes.
        server = await self.setup_test_server_single_model_instance()
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        # Mock a call that should result in skipping the round.
        await server.evaluation_round(
            round_i=1,
            valid_cfg=[EvaluateConfig(batch_size=8, frequency=2)],
            clients=server.netwk.client_names,
        )
        # Assert that no message was sent (routine was skipped).
        server.netwk.broadcast_message.assert_not_called()
        server.netwk.send_messages.assert_not_called()
        server.netwk.send_message.assert_not_called()

    @pytest.mark.parametrize("secagg", [False, True], ids=["clrtxt", "secagg"])
    @pytest.mark.asyncio
    async def test_fairness_round(
        self,
        secagg: bool,
    ) -> None:
        """Test that the 'fairness_round' routine triggers expected calls."""
        # Set up a server with mocked attributes.
        server = await self.setup_test_server_single_model_instance(
            use_secagg=secagg, use_fairness=True
        )
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert isinstance(server.model, List)
        assert isinstance(server.model[0], mock.NonCallableMagicMock)

        assert isinstance(server.fairness, mock.NonCallableMagicMock)
        assert isinstance(server.ckptr, mock.NonCallableMagicMock)
        # Mock-run a fairness routine.
        await server.fairness_round(
            round_i=0,
            fairness_cfg=FairnessConfig(),
            clients=server.netwk.client_names,
        )
        # Verify that expected actions occured.
        # (a) optional secagg setup
        decrypter: Optional[Decrypter] = None
        if secagg:
            assert isinstance(server.secagg, mock.NonCallableMagicMock)
            server.secagg.setup_decrypter.assert_awaited_once()
            decrypter = server.secagg.setup_decrypter.return_value
        # (b) fairness query emission, including model weights
        server.netwk.send_messages.assert_awaited_once()
        queries = server.netwk.send_messages.await_args[0][0]
        assert isinstance(queries, dict)
        assert queries.keys() == server.netwk.client_names
        for query in queries.values():
            assert isinstance(query, MultiModelFairnessQuery)
            for request in query.requests:
                assert (
                    request.weights is server.model[0].get_weights.return_value
                )
        # (c) fairness controller round routine
        server.fairness.run_fairness_round.assert_awaited_once_with(
            netwk=server.netwk, secagg=decrypter
        )
        # (d) checkpointing
        server.ckptr.save_metrics.assert_called_once_with(
            metrics=server.fairness.run_fairness_round.return_value,
            prefix="fairness_metrics",
            append=False,
            timestamp="round_0",
        )

    @pytest.mark.asyncio
    async def test_fairness_round_undefined(
        self,
    ) -> None:
        """Test that 'fairness_round' early-exits when fairness is not set."""
        # Set up a server with mocked attributes and no fairness controller.
        server = await self.setup_test_server_single_model_instance(
            use_fairness=False
        )
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert server.fairness is None
        # Call the fairness round routine.1
        await server.fairness_round(
            round_i=0,
            fairness_cfg=FairnessConfig(),
            clients=server.netwk.client_names,
        )
        # Assert that no message was sent (routine was skipped).
        server.netwk.broadcast_message.assert_not_called()
        server.netwk.send_messages.assert_not_called()
        server.netwk.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_fairness_round_skip(
        self,
    ) -> None:
        """Test that 'fairness_round' skips rounds when configured."""
        # Set up a server with a mocked fairness controller.
        server = await self.setup_test_server_single_model_instance(
            use_fairness=True
        )
        assert isinstance(server.fairness, mock.NonCallableMagicMock)
        # Mock a call that should result in skipping the round.
        await server.fairness_round(
            round_i=1,
            fairness_cfg=FairnessConfig(frequency=2),
            clients=server.netwk.client_names,
        )
        # Assert that the round was skipped.
        server.fairness.run_fairness_round.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_training(
        self,
    ) -> None:
        """Test that 'stop_training' triggers expected actions."""
        # Set up a server with mocked attributes.
        server = await self.setup_test_server_single_model_instance()
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert isinstance(server.model, List)

        assert isinstance(server.model[0], mock.NonCallableMagicMock)
        assert isinstance(server.ckptr, mock.NonCallableMagicMock)
        server.ckptr.folder = "mock_folder"
        # Call the 'stop_training' routine.
        await server.stop_training(rounds=5)
        # Verify that the expected message was broadcasted.
        server.netwk.broadcast_message.assert_awaited_once()
        message = server.netwk.broadcast_message.await_args[0][0]
        assert isinstance(message, StopTraining)
        assert (
            message.weights["model_0"]
            is server.model[0].get_weights.return_value
        )
        assert math.isnan(message.loss["model_0"])
        assert message.rounds == 5
        # Verify that the expected checkpointing occured.
        server.ckptr.save_model.assert_called_once_with(
            server.model[0], timestamp="best", idx=None
        )


class TestFederatedServerRun:
    """Unit tests for 'FederatedServer.run' and 'async_run' routines."""

    # Unit tests for FLRunConfig parsing via synchronous 'run' method.

    def test_run_from_dict(
        self,
    ) -> None:
        """Test that 'run' properly parses input dict config.

        Mock the actual underlying routine.
        """
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=MOCK_OPTIM
        )
        config = mock.create_autospec(dict, instance=True)
        with mock.patch.object(
            FLRunConfig,
            "from_params",
            return_value=mock.create_autospec(FLRunConfig, instance=True),
        ) as patch_flrunconfig_from_params:
            with mock.patch.object(server, "async_run") as patch_async_run:
                server.run(config)
        patch_flrunconfig_from_params.assert_called_once_with(**config)
        patch_async_run.assert_called_once_with(
            patch_flrunconfig_from_params.return_value
        )

    def test_run_from_toml(
        self,
    ) -> None:
        """Test that 'run' properly parses input TOML file.

        Mock the actual underlying routine.
        """
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=MOCK_OPTIM
        )
        config = "mock_path.toml"
        with mock.patch.object(
            FLRunConfig,
            "from_toml",
            return_value=mock.create_autospec(FLRunConfig, instance=True),
        ) as patch_flrunconfig_from_toml:
            with mock.patch.object(server, "async_run") as patch_async_run:
                server.run(config)
        patch_flrunconfig_from_toml.assert_called_once_with(config)
        patch_async_run.assert_called_once_with(
            patch_flrunconfig_from_toml.return_value
        )

    def test_run_from_config(
        self,
    ) -> None:
        """Test that 'run' properly uses input FLRunConfig.

        Mock the actual underlying routine.
        """
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=MOCK_OPTIM
        )
        config = mock.create_autospec(FLRunConfig, instance=True)
        with mock.patch.object(server, "async_run") as patch_async_run:
            server.run(config)
        patch_async_run.assert_called_once_with(config)

    # Unit tests for overall actions sequence in 'async_run'.

    @pytest.mark.asyncio
    async def test_async_run_actions_sequence(self) -> None:
        """Test that 'async_run' triggers expected routines."""
        # Setup a server and a run config with mock attributes.
        server = FederatedServer(
            model=MOCK_MODEL,
            netwk=MOCK_NETWK,
            optim=MOCK_OPTIM,
            checkpoint=mock.create_autospec(Checkpointer, instance=True),
        )
        config = FLRunConfig(
            rounds=10,
            register=mock.create_autospec(RegisterConfig, instance=True),
            training=TrainingConfig(8),
            evaluate=EvaluateConfig(8),
            fairness=mock.create_autospec(FairnessConfig, instance=True),
            privacy=None,
            early_stop=None,
        )
        # Call 'async_run', mocking all underlying routines.
        with mock.patch.object(
            server, "initialization"
        ) as patch_initialization:
            with mock.patch.object(server, "training_round") as patch_training:
                with mock.patch.object(
                    server, "evaluation_round"
                ) as patch_evaluation:
                    with mock.patch.object(
                        server, "fairness_round"
                    ) as patch_fairness:
                        with mock.patch.object(
                            server, "stop_training"
                        ) as patch_stop_training:
                            await server.async_run(config)
        # Verify that expected calls occured.
        patch_initialization.assert_called_once_with(config)
        patch_training.assert_has_calls(
            [
                mock.call(idx, config.training, server.netwk.client_names)
                for idx in range(1, 11)
            ]
        )
        patch_evaluation.assert_has_calls(
            [
                mock.call(idx, config.evaluate, server.netwk.client_names)
                for idx in range(1, 11)
            ]
        )
        patch_fairness.assert_has_calls(
            [
                mock.call(idx, config.fairness, server.netwk.client_names)
                for idx in range(0, 10)
            ]
            + [
                mock.call(
                    10,
                    config.fairness,
                    server.netwk.client_names,
                    force_run=True,
                )
            ]
        )
        patch_stop_training.assert_called_once_with(10)
