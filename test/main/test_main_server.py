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

"""Unit tests for 'FederatedServer'."""

import logging
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
    EvaluationRequest,
    FairnessQuery,
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
    TrainRequest,
)
from declearn.metrics import MetricSet
from declearn.model.api import Model
from declearn.model.sklearn import SklearnSGDModel
from declearn.optimizer import Optimizer
from declearn.secagg.api import Decrypter, SecaggConfigServer
from declearn.secagg.messaging import (
    SecaggEvaluationReply,
    SecaggTrainReply,
)
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
            fairness=mock.create_autospec(
                FairnessControllerServer, instance=True
            ),
        )
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=optim
        )
        assert server.c_opt is optim.client_opt
        assert server.optim is optim.server_opt
        assert server.aggrg is optim.aggregator
        assert server.fairness is optim.fairness

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
        server = FederatedServer(
            model=MOCK_MODEL, netwk=MOCK_NETWK, optim=optim
        )
        assert server.c_opt is optim["client_opt"]
        assert server.optim is optim["server_opt"]
        assert server.aggrg is optim["aggregator"]
        assert server.fairness is optim["fairness"]

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
        assert server.metrics is metrics

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


class TestFederatedServerRoutines:
    """Unit tests for 'FederatedServer' main unitary routines."""

    @staticmethod
    async def setup_test_server(
        use_secagg: bool = False,
        use_fairness: bool = False,
    ) -> FederatedServer:
        """Set up a FederatedServer wrapping mock controllers."""
        netwk = mock.create_autospec(NetworkServer, instance=True)
        netwk.name = "server"
        netwk.client_names = {"client_a", "client_b"}
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
        secagg: Optional[SecaggConfigServer] = None
        if use_secagg:
            secagg = mock.create_autospec(SecaggConfigServer, instance=True)
            secagg.secagg_type = "mock_secagg"  # type: ignore
        return FederatedServer(
            model=mock.create_autospec(Model, instance=True),
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
        """Test that the 'initialization' routine triggers expected calls."""
        # Set up a server with mocked attributes.
        server = await self.setup_test_server(
            use_secagg=secagg, use_fairness=fairness
        )
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert isinstance(server.model, mock.NonCallableMagicMock)
        server.model.required_data_info = {"n_samples"} if metadata else {}
        aggrg = server.aggrg
        # Run the initialization routine.
        config = FLRunConfig.from_params(
            rounds=10,
            register=RegisterConfig(0, 2, 120),
            training={"batch_size": 8},
            privacy=(
                {"budget": (1e-3, 0.0), "sclip_norm": 1.0} if privacy else None
            ),
        )
        server.netwk.wait_for_messages.side_effect = self._setup_init_replies(
            metadata, privacy
        )
        await server.initialization(config)
        # Verify that the clients-registration routine was called.
        server.netwk.wait_for_clients.assert_awaited_once_with(0, 2, 120)
        # Verify that the expected number of message exchanges occured.
        assert server.netwk.broadcast_message.await_count == (
            1 + metadata + privacy
        )
        queries = server.netwk.broadcast_message.await_args_list.copy()
        # When configured, verify that metadata were queried and used.
        if metadata:
            query = queries.pop(0)[0][0]
            assert isinstance(query, MetadataQuery)
            assert query.fields == ["n_samples"]
            server.model.initialize.assert_called_once_with({"n_samples": 200})
        # Verify that an InitRequest was sent with expected parameters.
        query = queries.pop(0)[0][0]
        assert isinstance(query, InitRequest)
        assert query.dpsgd is privacy
        if secagg:
            assert query.secagg is not None
        else:
            assert query.secagg is None
        assert query.fairness is fairness
        # Verify that DP-SGD setup occurred when expected.
        if privacy:
            query = queries.pop(0)[0][0]
            assert isinstance(query, PrivacyRequest)
            assert query.budget == (1e-3, 0.0)
            assert query.sclip_norm == 1.0
            assert query.rounds == 10
        # Verify that SecAgg setup occurred when expected.
        decrypter: Optional[Decrypter] = None
        if secagg:
            assert isinstance(server.secagg, mock.NonCallableMagicMock)
            if fairness:
                server.secagg.setup_decrypter.assert_awaited_once()
                decrypter = server.secagg.setup_decrypter.return_value
            else:
                server.secagg.setup_decrypter.assert_not_called()
        # Verify that fairness setup occurred when expected.
        if fairness:
            assert isinstance(server.fairness, mock.NonCallableMagicMock)
            server.fairness.setup_fairness.assert_awaited_once_with(
                netwk=server.netwk, aggregator=aggrg, secagg=decrypter
            )
            assert server.aggrg is server.fairness.setup_fairness.return_value

    def _setup_init_replies(
        self,
        metadata: bool,
        privacy: bool,
    ) -> List[Dict[str, mock.NonCallableMagicMock]]:
        clients = ("client_a", "client_b")
        messages: List[Dict[str, mock.NonCallableMagicMock]] = []
        if metadata:
            msg = MetadataReply({"n_samples": 100})
            messages.append(
                {
                    key: self.setup_mock_serialized_message(MetadataReply, msg)
                    for key in clients
                }
            )
        messages.append(
            {
                key: self.setup_mock_serialized_message(InitReply)
                for key in clients
            }
        )
        if privacy:
            messages.append(
                {
                    key: self.setup_mock_serialized_message(PrivacyReply)
                    for key in clients
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
        server = await self.setup_test_server(use_secagg=secagg)
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert isinstance(server.model, mock.NonCallableMagicMock)
        assert isinstance(server.optim, mock.NonCallableMagicMock)
        assert isinstance(server.aggrg, mock.NonCallableMagicMock)
        # Mock-run a training routine.
        reply_cls: Type[Message] = (
            SecaggTrainReply if secagg else TrainReply  # type: ignore
        )
        updates = mock.create_autospec(ModelUpdates, instance=True)
        reply_msg = TrainReply(
            n_epoch=1, n_steps=10, t_spent=0.0, updates=updates, aux_var={}
        )
        wrapped = None if secagg else reply_msg
        server.netwk.wait_for_messages.return_value = {
            "client_a": self.setup_mock_serialized_message(reply_cls, wrapped),
            "client_b": self.setup_mock_serialized_message(reply_cls, wrapped),
        }
        with mock.patch(
            "declearn.secagg.messaging.aggregate_secagg_messages",
            return_value=reply_msg,
        ) as patch_aggregate_secagg_messages:
            await server.training_round(
                round_i=1, train_cfg=TrainingConfig(batch_size=8)
            )
        # Verify that expected actions occured.
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
            assert isinstance(query, TrainRequest)
            assert query.weights is server.model.get_weights.return_value
            assert query.aux_var is server.optim.collect_aux_var.return_value
        # (c) training reply reception
        server.netwk.wait_for_messages.assert_awaited_once()
        if secagg:
            patch_aggregate_secagg_messages.assert_called_once()
        else:
            patch_aggregate_secagg_messages.assert_not_called()
        # (d) updates aggregation and global model weights update
        server.optim.process_aux_var.assert_called_once()
        server.aggrg.finalize_updates.assert_called_once()
        server.optim.apply_gradients.assert_called_once()

    @pytest.mark.parametrize("secagg", [False, True], ids=["clrtxt", "secagg"])
    @pytest.mark.asyncio
    async def test_evaluation_round(
        self,
        secagg: bool,
    ) -> None:
        """Test that the 'evaluation_round' routine triggers expected calls."""
        # Set up a server with mocked attributes.
        server = await self.setup_test_server(use_secagg=secagg)
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert isinstance(server.model, mock.NonCallableMagicMock)
        assert isinstance(server.metrics, mock.NonCallableMagicMock)
        assert isinstance(server.ckptr, mock.NonCallableMagicMock)
        # Mock-run an evaluation routine.
        reply_cls: Type[Message] = (
            SecaggEvaluationReply  # type: ignore
            if secagg
            else EvaluationReply
        )
        reply_msg = EvaluationReply(
            loss=0.42, n_steps=10, t_spent=0.0, metrics={}
        )
        wrapped = None if secagg else reply_msg
        server.netwk.wait_for_messages.return_value = {
            "client_a": self.setup_mock_serialized_message(reply_cls, wrapped),
            "client_b": self.setup_mock_serialized_message(reply_cls, wrapped),
        }
        with mock.patch(
            "declearn.secagg.messaging.aggregate_secagg_messages",
            return_value=reply_msg,
        ) as patch_aggregate_secagg_messages:
            await server.evaluation_round(
                round_i=1, valid_cfg=EvaluateConfig(batch_size=8)
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
            assert isinstance(query, EvaluationRequest)
            assert query.weights is server.model.get_weights.return_value
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
            model=server.model,
            optimizer=server.optim,
            metrics=server.metrics.get_result.return_value,
        )

    @pytest.mark.asyncio
    async def test_evaluation_round_skip(
        self,
    ) -> None:
        """Test that 'evaluation_round' skips rounds when configured."""
        # Set up a server with mocked attributes.
        server = await self.setup_test_server()
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        # Mock a call that should result in skipping the round.
        await server.evaluation_round(
            round_i=1,
            valid_cfg=EvaluateConfig(batch_size=8, frequency=2),
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
        server = await self.setup_test_server(
            use_secagg=secagg, use_fairness=True
        )
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert isinstance(server.model, mock.NonCallableMagicMock)
        assert isinstance(server.fairness, mock.NonCallableMagicMock)
        assert isinstance(server.ckptr, mock.NonCallableMagicMock)
        # Mock-run a fairness routine.
        await server.fairness_round(
            round_i=0,
            fairness_cfg=FairnessConfig(),
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
            assert isinstance(query, FairnessQuery)
            assert query.weights is server.model.get_weights.return_value
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
        server = await self.setup_test_server(use_fairness=False)
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert server.fairness is None
        # Call the fairness round routine.1
        await server.fairness_round(
            round_i=0,
            fairness_cfg=FairnessConfig(),
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
        server = await self.setup_test_server(use_fairness=True)
        assert isinstance(server.fairness, mock.NonCallableMagicMock)
        # Mock a call that should result in skipping the round.
        await server.fairness_round(
            round_i=1,
            fairness_cfg=FairnessConfig(frequency=2),
        )
        # Assert that the round was skipped.
        server.fairness.run_fairness_round.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_training(
        self,
    ) -> None:
        """Test that 'stop_training' triggers expected actions."""
        # Set up a server with mocked attributes.
        server = await self.setup_test_server()
        assert isinstance(server.netwk, mock.NonCallableMagicMock)
        assert isinstance(server.model, mock.NonCallableMagicMock)
        assert isinstance(server.ckptr, mock.NonCallableMagicMock)
        server.ckptr.folder = "mock_folder"
        # Call the 'stop_training' routine.
        await server.stop_training(rounds=5)
        # Verify that the expected message was broadcasted.
        server.netwk.broadcast_message.assert_awaited_once()
        message = server.netwk.broadcast_message.await_args[0][0]
        assert isinstance(message, StopTraining)
        assert message.weights is server.model.get_weights.return_value
        assert math.isnan(message.loss)
        assert message.rounds == 5
        # Verify that the expected checkpointing occured.
        server.ckptr.save_model.assert_called_once_with(
            server.model, timestamp="best"
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
            training=mock.create_autospec(TrainingConfig, instance=True),
            evaluate=mock.create_autospec(EvaluateConfig, instance=True),
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
            [mock.call(idx, config.training) for idx in range(1, 11)]
        )
        patch_evaluation.assert_has_calls(
            [mock.call(idx, config.evaluate) for idx in range(1, 11)]
        )
        patch_fairness.assert_has_calls(
            [mock.call(idx, config.fairness) for idx in range(0, 10)]
            + [mock.call(10, config.fairness, force_run=True)]
        )
        patch_stop_training.assert_called_once_with(10)
