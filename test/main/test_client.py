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

"""Unit tests for 'FederatedClient'."""

import logging
from unittest import mock

import pytest  # type: ignore

from declearn import messaging
from declearn.dataset import Dataset
from declearn.communication import NetworkClientConfig
from declearn.communication.api import NetworkClient
from declearn.main import FederatedClient
from declearn.main.utils import Checkpointer, TrainingManager
from declearn.secagg.api import SecaggConfigClient, SecaggSetupQuery
from declearn.secagg.masking.messages import MaskingSecaggSetupQuery
from declearn.utils import LOGGING_LEVEL_MAJOR


MOCK_NETWK = mock.create_autospec(NetworkClient, instance=True)
MOCK_NETWK.name = "client"
MOCK_DATASET = mock.create_autospec(Dataset, instance=True)


class TestFederatedClientInit:  # pylint: disable=too-many-public-methods
    """Unit tests for 'FederatedClient.__init__'."""

    # Tests for the 'netwk' argument.

    def test_netwk_instance(self) -> None:
        """Test specifying 'netwk' as a 'NetworkClient' instance."""
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        assert client.netwk is netwk

    def test_netwk_config(self) -> None:
        """Test specifying 'netwk' as a 'NetworkClientConfig' instance."""
        netwk = mock.create_autospec(NetworkClientConfig, instance=True)
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        netwk.build_client.assert_called_once()
        assert client.netwk is netwk.build_client.return_value

    def test_netwk_config_dict(self) -> None:
        """Test specifying 'netwk' as a properly-parsable dict."""
        netwk = {"protocol": "mock", "server_uri": "uri", "name": "name"}
        with mock.patch.object(NetworkClientConfig, "build_client") as patched:
            client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        patched.assert_called_once()
        assert client.netwk is patched.return_value

    def test_netwk_config_file(self) -> None:
        """Test specifying 'netwk' as a path to a TOML file."""
        netwk = "stub_path_to_netwk_config.toml"
        with mock.patch.object(NetworkClientConfig, "from_toml") as patched:
            client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        patched.assert_called_once_with(netwk)
        patched.return_value.build_client.assert_called_once()
        assert client.netwk is patched.return_value.build_client.return_value

    def test_netwk_config_invalid(self) -> None:
        """Test specifying 'netwk' as an invalid type."""
        with pytest.raises(TypeError):
            FederatedClient(netwk=mock.MagicMock(), train_data=MOCK_DATASET)

    # Tests for the 'train_data' argument.

    def test_train_data_instance(self) -> None:
        """Test specifying 'train_data' as a Dataset."""
        dataset = mock.create_autospec(Dataset, instance=True)
        client = FederatedClient(netwk=MOCK_NETWK, train_data=dataset)
        assert client.train_data is dataset

    def test_train_data_str(self) -> None:
        """Test specifying 'train_data' as a file path."""
        path = "mock_path_to_dataset.json"
        with mock.patch(
            "declearn.main._client.load_dataset_from_json",
            return_value=mock.create_autospec(Dataset, instance=True),
        ) as patched:
            client = FederatedClient(netwk=MOCK_NETWK, train_data=path)
        patched.assert_called_once_with(path)
        assert client.train_data is patched.return_value

    def test_train_data_invalid(self) -> None:
        """Test specifying 'train_data' as an invalid type."""
        with pytest.raises(TypeError):
            FederatedClient(netwk=MOCK_NETWK, train_data=mock.MagicMock())

    # Tests for the 'valid_data' argument.

    def test_valid_data_none(self) -> None:
        """Test specifying 'valid_data' as None."""
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, valid_data=None
        )
        assert client.valid_data is None

    def test_valid_data_instance(self) -> None:
        """Test specifying 'valid_data' as a Dataset."""
        dataset = mock.create_autospec(Dataset, instance=True)
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, valid_data=dataset
        )
        assert client.valid_data is dataset

    def test_valid_data_str(self) -> None:
        """Test specifying 'valid_data' as a file path."""
        path = "mock_path_to_dataset.json"
        with mock.patch(
            "declearn.main._client.load_dataset_from_json",
            return_value=mock.create_autospec(Dataset, instance=True),
        ) as patched:
            client = FederatedClient(
                netwk=MOCK_NETWK, train_data=MOCK_DATASET, valid_data=path
            )
        patched.assert_called_once_with(path)
        assert client.valid_data is patched.return_value

    def test_valid_data_invalid(self) -> None:
        """Test specifying 'valid_data' as an invalid type."""
        with pytest.raises(TypeError):
            FederatedClient(
                netwk=MOCK_NETWK,
                train_data=MOCK_DATASET,
                valid_data=mock.MagicMock(),
            )

    # Tests for the 'checkpoint' argument.

    def test_checkpoint_instance(self) -> None:
        """Test specifying 'checkpoint' as a Checkpointer instance."""
        checkpointer = mock.create_autospec(Checkpointer, instance=True)
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, checkpoint=checkpointer
        )
        assert client.ckptr is checkpointer

    def test_checkpoint_none(self) -> None:
        """Test specifying 'checkpoint' as None."""
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, checkpoint=None
        )
        assert client.ckptr is None

    def test_checkpoint_specs(self) -> None:
        """Test specifying 'checkpoint' as some specs.

        Note: 'Checkpointer.from_specs' has its own unit tests for subcases.
        """
        specs = {"folder": "mock_folder", "max_history": 1}
        with mock.patch.object(Checkpointer, "from_specs") as patched:
            client = FederatedClient(
                netwk=MOCK_NETWK, train_data=MOCK_DATASET, checkpoint=specs
            )
        patched.assert_called_once_with(specs)
        assert client.ckptr is patched.return_value

    # Tests for the 'secagg' argument.

    def test_secagg_instance(self) -> None:
        """Test specifying 'secagg' as a SecaggConfigClient instance."""
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, secagg=secagg
        )
        assert client.secagg is secagg

    def test_secagg_none(self) -> None:
        """Test specifying 'secagg' as None."""
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, secagg=None
        )
        assert client.secagg is None

    def test_secagg_dict(self) -> None:
        """Test specifying 'secagg' as a config dict."""
        secagg = {"secagg_type": "mock", "id_keys": mock.MagicMock()}
        with mock.patch(
            "declearn.main._client.parse_secagg_config_client"
        ) as patched:
            client = FederatedClient(
                netwk=MOCK_NETWK, train_data=MOCK_DATASET, secagg=secagg
            )
        patched.assert_called_once_with(**secagg)
        assert client.secagg is patched.return_value

    def test_secagg_invalid(self) -> None:
        """Test specifying 'secagg' as an invalid type."""
        with pytest.raises(TypeError):
            FederatedClient(
                netwk=MOCK_NETWK,
                train_data=MOCK_DATASET,
                secagg=mock.MagicMock(),
            )

    # Tests for the 'share_metrics' argument.

    def test_share_metrics(self) -> None:
        """Test that 'share_metrics' is properly attributed."""
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, share_metrics=True
        )
        assert client.share_metrics is True
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, share_metrics=False
        )
        assert client.share_metrics is False

    # Tests for the 'logger' argument.

    def test_logger_instance(self) -> None:
        """Test specifying 'logger' as a Logger instance."""
        logger = logging.Logger("mock-client-logger")
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, logger=logger
        )
        assert client.logger is logger

    def test_logger_str(self) -> None:
        """Test specifying 'logger' as a logger name."""
        logger = "mock-client-logger"
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, logger=logger
        )
        assert isinstance(client.logger, logging.Logger)
        assert client.logger.name == logger

    def test_logger_none(self) -> None:
        """Test specifying 'logger' as None."""
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, logger=None
        )
        assert isinstance(client.logger, logging.Logger)

    def test_logger_invalid(self) -> None:
        """Test specifying 'logger' with a wrong type."""
        with pytest.raises(TypeError):
            FederatedClient(
                netwk=MOCK_NETWK,
                train_data=MOCK_DATASET,
                logger=mock.MagicMock(),
            )

    # Tests for the 'verbose' argument.

    def test_verbose_true(self) -> None:
        """Test that 'verbose=True' is recorded and sets proper logging."""
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, verbose=True
        )
        assert client.verbose is True
        assert client.logger.level is logging.INFO

    def test_verbose_false(self) -> None:
        """Test that 'verbose=False' is recorded and sets proper logging."""
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, verbose=False
        )
        assert client.verbose is False
        assert client.logger.level is LOGGING_LEVEL_MAJOR


class TestFederatedClient:
    """Unit tests for some points of behavior of 'FederatedClient'."""

    @pytest.mark.asyncio
    async def test_register_failure(self) -> None:
        """Test that a RuntimeError is raised when registration fails.

        Also verify that 10 attempts with a (bypassed) 1-minute delay are done.
        """
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        netwk.register.return_value = False
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        with mock.patch("asyncio.sleep", new=mock.AsyncMock()) as patched:
            with pytest.raises(RuntimeError):
                await client.register()
        netwk.register.assert_has_awaits([mock.call()] * 10)
        patched.assert_has_awaits([mock.call(60)] * 10)

    @pytest.mark.asyncio
    async def test_initialize_failure_secagg_mismatch(self) -> None:
        """Test that an InitRequest with mismatching secagg raises an error."""
        # Set up a client with a mock network that will receive an InitRequest.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        msg = mock.create_autospec(messaging.SerializedMessage, instance=True)
        msg.message_cls = messaging.InitRequest
        msg.deserialize.return_value = messaging.InitRequest(
            model=mock.MagicMock(),
            optim=mock.MagicMock(),
            aggrg=mock.MagicMock(),
            secagg="mock-secagg",
        )
        netwk.recv_message.return_value = msg
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Attempt running initialization, monitoring TrainingManager.
        with mock.patch.object(TrainingManager, "__init__") as patched:
            with pytest.raises(RuntimeError):
                await client.initialize()
        # Assert that an Error was sent to the server and TrainingManager
        # instantiation was not even attempted.
        netwk.send_message.assert_called_once()
        assert isinstance(netwk.send_message.call_args[0][0], messaging.Error)
        patched.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_secagg_no_secagg(self) -> None:
        """Test that 'setup_secagg' fails if no SecAgg is configured."""
        # Set up a client with mock NetworkClient.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Test that a SecAgg setup query trigger an Error reply.
        msg = messaging.SerializedMessage.from_message_string(
            MaskingSecaggSetupQuery(bitsize=32, clipval=10.0).to_string()
        )  # type: messaging.SerializedMessage[SecaggSetupQuery]
        await client.setup_secagg(msg)
        netwk.send_message.assert_called_once()
        assert isinstance(netwk.send_message.call_args[0][0], messaging.Error)

    @pytest.mark.asyncio
    async def test_setup_secagg_error_catching(self) -> None:
        """Test that SecAgg setup errors within 'setup_secagg' are caught."""
        # Set up a client with mock NetworkClient and SecaggConfigClient.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        client = FederatedClient(
            netwk=netwk, train_data=MOCK_DATASET, secagg=secagg
        )
        secagg.setup_encrypter.side_effect = ValueError
        # Test that the exception is caught.
        msg = messaging.SerializedMessage.from_message_string(
            MaskingSecaggSetupQuery(bitsize=32, clipval=10.0).to_string()
        )  # type: messaging.SerializedMessage[SecaggSetupQuery]
        await client.setup_secagg(msg)
        secagg.setup_encrypter.assert_awaited_once_with(netwk=netwk, query=msg)
        netwk.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_share_metrics(self) -> None:
        """Test that 'share_metrics' has expected effect on eval. replies."""

        def setup(share_metrics: bool):
            """Set up a FederatedClient with a mock TrainingManager."""
            client = FederatedClient(
                netwk=mock.create_autospec(NetworkClient, instance=True),
                train_data=mock.create_autospec(Dataset, instance=True),
                logger="mock-client-logger",
                share_metrics=share_metrics,
            )
            manager = mock.create_autospec(TrainingManager, instance=True)
            manager.evaluation_round.return_value = messaging.EvaluationReply(
                loss=0.42,
                n_steps=10,
                t_spent=4.2,
                metrics={"metrics": mock.MagicMock()},
            )
            client.trainmanager = manager
            return client

        # Run with 'share_metrics=True' and verify sent reply.
        client = setup(share_metrics=True)
        query = mock.create_autospec(messaging.TrainReply, instance=True)
        await client.evaluation_round(query)
        client.trainmanager.evaluation_round.assert_called_once_with(query)
        reply = client.trainmanager.evaluation_round.return_value
        client.netwk.send_message.assert_called_once_with(reply)

        # Run with 'share_metrics=False' and verify sent reply.
        client = setup(share_metrics=False)
        query = mock.create_autospec(messaging.TrainReply, instance=True)
        await client.evaluation_round(query)
        client.trainmanager.evaluation_round.assert_called_once_with(query)
        reply = client.trainmanager.evaluation_round.return_value
        reply.metrics = {}
        client.netwk.send_message.assert_called_once_with(reply)
