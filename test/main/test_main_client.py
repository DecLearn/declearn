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

"""Unit tests for 'FederatedClient'."""

import contextlib
import logging
from typing import Any, Iterator, Optional, Tuple, Type
from unittest import mock

import pytest  # type: ignore

from declearn import messaging
from declearn.communication import NetworkClientConfig
from declearn.communication.api import NetworkClient
from declearn.dataset import Dataset, DataSpecs
from declearn.fairness.api import FairnessControllerClient
from declearn.main import FederatedClient
from declearn.main.utils import Checkpointer
from declearn.metrics import MetricState
from declearn.model.api import Model
from declearn.secagg import messaging as secagg_messaging
from declearn.secagg.api import SecaggConfigClient, SecaggSetupQuery
from declearn.training import TrainingManager
from declearn.utils import LOGGING_LEVEL_MAJOR

try:
    from declearn.training.dp import DPTrainingManager
except ModuleNotFoundError:
    DP_AVAILABLE = False
else:
    DP_AVAILABLE = True


# numerous but organized tests; pylint: disable=too-many-lines


MOCK_NETWK = mock.create_autospec(NetworkClient, instance=True)
MOCK_NETWK.name = "client"
MOCK_DATASET = mock.create_autospec(Dataset, instance=True)


def object_new(cls, *_, **__) -> Any:
    """Wrapper for 'object.__new__' accepting/discarding *args and **kwargs."""
    return object.__new__(cls)


@contextlib.contextmanager
def patch_class_constructor(
    cls: Type[Any],
    **kwargs: Any,
) -> Iterator[mock.Mock]:
    """Patch a class constructor (its '__new__' method).

    Overload `unittest.mock.patch.object(cls, '__new__', **kwargs)`
    to properly restore the initial `__new__` method's behavior when
    it is the base `object.__new__` one.

    This function is adapted from the following StackOverflow post:
    https://stackoverflow.com/questions/65360692/python-patching-new-method

    NOTE: when patching multiple classes, if class A inherits class B,
    then B **must** be patched **before** A so that they are properly
    reset to (replacements of) `object.__new__` at the end. Otherwise,
    class B will retain a mocked `__new__` method, and not just in the
    initial calling scope.
    """
    new = cls.__new__
    try:
        with mock.patch.object(cls, "__new__", **kwargs) as patch:
            yield patch
    finally:
        if new is object.__new__:
            cls.__new__ = object_new  # type: ignore[assignment]
        else:
            cls.__new__ = new


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

    def test_secagg_invalid_dict(self) -> None:
        """Test specifying 'secagg' as an invalid type."""
        secagg = {"secagg_type": "mock", "id_keys": mock.MagicMock()}
        with pytest.raises(TypeError):
            FederatedClient(
                netwk=MOCK_NETWK, train_data=MOCK_DATASET, secagg=secagg
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

    def test_share_metrics_disabled_with_secagg(self) -> None:
        """Test that 'share_metrics=False' with SecAgg emits a warning."""
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        with pytest.warns(UserWarning):
            FederatedClient(
                MOCK_NETWK, MOCK_DATASET, share_metrics=False, secagg=secagg
            )

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


class TestFederatedClientInitialize:
    """Unit tests for 'FederatedClient.initialize'.

    Theses tests emulate a large number of scenarios, notably focusing
    on potential errors to verify the error catching and raising hooks.

    They rely on mocks and patching rather than actual setup, which is
    somewhat tedious when reviewing the tests, but enables pinpointing
    error types and causes and avoids any costly setup to run tests.
    """

    @staticmethod
    def _setup_mock_init_request(
        secagg: Optional[str] = None,
        dpsgd: bool = False,
        fairness: bool = False,
    ) -> messaging.SerializedMessage[messaging.InitRequest]:
        """Return a mock serialized InitRequest."""
        init_req = messaging.InitRequest(
            model=mock.MagicMock(),
            optim=mock.MagicMock(),
            aggrg=mock.MagicMock(),
            secagg=secagg,
            dpsgd=dpsgd,
            fairness=fairness,
        )
        msg_init = mock.create_autospec(
            messaging.SerializedMessage, instance=True
        )
        msg_init.message_cls = messaging.InitRequest
        msg_init.deserialize.return_value = init_req
        return msg_init

    @pytest.mark.asyncio
    async def test_initialize_simple(self) -> None:
        """Test that initialization with a single request works as expected."""
        # Set up amock network endpoint receiving an InitRequest.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        netwk.recv_message.return_value = self._setup_mock_init_request()
        # Set up a client with that endpoint.
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Attempt running initialization, monitoring TrainingManager.
        with patch_class_constructor(TrainingManager) as patch_tm:
            await client.initialize()
        # Assert that an InitReply was sent to the server.
        netwk.send_message.assert_called_once_with(messaging.InitReply())
        # Assert that a TrainingManager was instantiated and assigned.
        patch_tm.assert_called_once()
        assert client.trainmanager is patch_tm.return_value

    @pytest.mark.asyncio
    async def test_initialize_error_catching(self) -> None:
        """Test that initialization error are properly handled."""
        # Set up amock network endpoint receiving an InitRequest.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        netwk.recv_message.return_value = self._setup_mock_init_request()
        # Set up a client with that endpoint.
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Attempt running initialization, forcing TrainingManager init failure.
        with patch_class_constructor(TrainingManager) as patch_tm:
            patch_tm.side_effect = TypeError
            with pytest.raises(RuntimeError):
                await client.initialize()
        patch_tm.assert_called_once()
        # Assert that an Error was sent to the server.
        netwk.send_message.assert_called_once()
        reply = netwk.send_message.call_args.args[0]
        assert isinstance(reply, messaging.Error)

    @pytest.mark.asyncio
    async def test_initialize_with_metadata_query(self) -> None:
        """Test that initialization with a MetadataQuery works as expected."""
        # Set up a mock network receiving a MetadataQuery and an InitRequest.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        msg_data: messaging.SerializedMessage[messaging.MetadataQuery] = (
            messaging.SerializedMessage.from_message_string(
                messaging.MetadataQuery(fields=["n_samples"]).to_string()
            )
        )
        msg_init = self._setup_mock_init_request()
        netwk.recv_message.side_effect = [msg_data, msg_init]
        # Set up a client with a mock dataset returning some arbitrary specs.
        dataset = mock.create_autospec(Dataset, instance=True)
        dataset.get_data_specs.return_value = DataSpecs(
            n_samples=100, features_shape=(32,)
        )
        client = FederatedClient(netwk=netwk, train_data=dataset)
        # Attempt running initialization, monitoring TrainingManager.
        with patch_class_constructor(TrainingManager) as patched:
            await client.initialize()
        # Assert that two replies were sent to the server.
        assert netwk.send_message.call_count == 2
        # Assert that a MetadataReply was first sent to the server.
        reply = netwk.send_message.call_args_list[0].args[0]
        assert isinstance(reply, messaging.MetadataReply)
        dataset.get_data_specs.assert_called_once()
        assert reply.data_info == {"n_samples": 100}
        # Assert that an InitReply was then sent to the server.
        reply = netwk.send_message.call_args_list[1].args[0]
        assert isinstance(reply, messaging.InitReply)
        # Assert that a TrainingManager was instantiated and assigned.
        patched.assert_called_once()
        assert client.trainmanager is patched.return_value

    @pytest.mark.asyncio
    async def test_initialize_with_metadata_query_error(self) -> None:
        """Test initialization with a malformed MetadataQuery."""
        # Set up a mock network receiving an invalid MetadataQuery.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        msg_data: messaging.SerializedMessage[messaging.MetadataQuery] = (
            messaging.SerializedMessage.from_message_string(
                messaging.MetadataQuery(fields=["invalid"]).to_string()
            )
        )
        netwk.recv_message.return_value = msg_data
        # Set up a client with a mock dataset returning some arbitrary specs.
        dataset = mock.create_autospec(Dataset, instance=True)
        dataset.get_data_specs.return_value = DataSpecs(
            n_samples=100, features_shape=(32,)
        )
        client = FederatedClient(netwk=netwk, train_data=dataset)
        # Attempt running initialization, that is expected to fail.
        with pytest.raises(RuntimeError):
            await client.initialize()
        # Assert that an Error was sent to the server.
        netwk.send_message.assert_called_once()
        reply = netwk.send_message.call_args.args[0]
        assert isinstance(reply, messaging.Error)
        # Assert that this happened after accessing data specs.
        dataset.get_data_specs.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_secagg(self) -> None:
        """Test that initialization with a single request works as expected."""
        # Set up amock network endpoint receiving an InitRequest with SecAgg.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        netwk.recv_message.return_value = self._setup_mock_init_request(
            secagg="mock-secagg", dpsgd=False
        )
        # Set up a client with that endpoint and a matching mock secagg.
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        secagg.secagg_type = "mock-secagg"
        client = FederatedClient(
            netwk=netwk, train_data=MOCK_DATASET, secagg=secagg
        )
        # Attempt running initialization, monitoring TrainingManager.
        with patch_class_constructor(TrainingManager) as patched:
            await client.initialize()
        # Assert that an InitReply was sent to the server.
        netwk.send_message.assert_called_once_with(messaging.InitReply())
        # Assert that a TrainingManager was instantiated and assigned.
        patched.assert_called_once()
        assert client.trainmanager is patched.return_value

    @pytest.mark.asyncio
    async def test_initialize_with_secagg_mismatch(self) -> None:
        """Test that an InitRequest with mismatching secagg raises an error."""
        # Set up amock network endpoint receiving an InitRequest with SecAgg.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        netwk.recv_message.return_value = self._setup_mock_init_request(
            secagg="mock-secagg", dpsgd=False
        )
        # Set up a client with that endpoint but a distinct secagg type.
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        secagg.secagg_type = "mock-secagg-bis"
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Attempt running initialization, monitoring TrainingManager.
        with patch_class_constructor(TrainingManager) as patched:
            with pytest.raises(RuntimeError):
                await client.initialize()
        # Assert that an Error was sent to the server and TrainingManager
        # instantiation was not even attempted.
        netwk.send_message.assert_called_once()
        reply = netwk.send_message.call_args.args[0]
        assert isinstance(reply, messaging.Error)
        patched.assert_not_called()

    def _setup_dpsgd_setup_query(
        self,
    ) -> Tuple[
        messaging.SerializedMessage[messaging.PrivacyRequest],
        messaging.PrivacyRequest,
    ]:
        """Setup a mock PrivacyRequest and a wrapping SerializedMessage."""
        dp_query = mock.create_autospec(
            messaging.PrivacyRequest, instance=True
        )
        msg_priv = mock.create_autospec(
            messaging.SerializedMessage, instance=True
        )
        msg_priv.message_cls = messaging.PrivacyRequest
        msg_priv.deserialize.return_value = dp_query
        return msg_priv, dp_query

    @pytest.mark.asyncio
    async def test_initialize_with_dpsgd(self) -> None:
        """Test that initialization with DP-SGD works properly."""
        if not DP_AVAILABLE:
            pytest.skip(reason="Unavailable DP features (missing Opacus).")
        # Set up a mock network receiving an InitRequest and a PrivacyRequest.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        msg_init = self._setup_mock_init_request(secagg=None, dpsgd=True)
        msg_priv, dp_query = self._setup_dpsgd_setup_query()
        netwk.recv_message.side_effect = [msg_init, msg_priv]
        # Set up a client wrapping the former network endpoint.
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Attempt running initialization, patching/monitoring both
        # TrainingManager and its DP counterpart.
        with patch_class_constructor(DPTrainingManager) as patch_dp:
            with patch_class_constructor(TrainingManager) as patch_tm:
                await client.initialize()
        # Assert that an InitReply and a PrivacyReply were sent to the server.
        assert netwk.send_message.call_count == 2
        reply = netwk.send_message.call_args_list[0].args[0]
        assert isinstance(reply, messaging.InitReply)
        reply = netwk.send_message.call_args_list[1].args[0]
        assert isinstance(reply, messaging.PrivacyReply)
        # Assert that a DPTrainingManager was set up.
        patch_tm.assert_called_once()
        patch_dp.assert_called_once_with(
            DPTrainingManager,
            model=patch_tm.return_value.model,
            optim=patch_tm.return_value.optim,
            aggrg=patch_tm.return_value.aggrg,
            train_data=patch_tm.return_value.train_data,
            valid_data=patch_tm.return_value.valid_data,
            metrics=patch_tm.return_value.metrics,
            logger=patch_tm.return_value.logger,
            verbose=patch_tm.return_value.verbose,
        )
        patch_dp.return_value.make_private.assert_called_once_with(dp_query)
        assert client.trainmanager is patch_dp.return_value

    @pytest.mark.asyncio
    async def test_initialize_with_dpsgd_error_wrong_message(self) -> None:
        """Test error catching for DP-SGD setup with wrong second message."""
        if not DP_AVAILABLE:
            pytest.skip(reason="Unavailable DP features (missing Opacus).")
        # Set up a mock network receiving a DP InitRequest but wrong follow-up.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        msg_init = self._setup_mock_init_request(secagg=None, dpsgd=True)
        netwk.recv_message.return_value = msg_init
        # Set up a client wrapping the former network endpoint.
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Attempt running initialization, patching/monitoring both
        # TrainingManager and its DP counterpart. Expect it to fail.
        with patch_class_constructor(DPTrainingManager) as patch_dp:
            with patch_class_constructor(TrainingManager) as patch_tm:
                with pytest.raises(RuntimeError):
                    await client.initialize()
        # Assert that two messages were fetched, that first step went well
        # (resulting in an InitReply) and then an Error was sent.
        assert netwk.recv_message.call_count == 2
        assert netwk.send_message.call_count == 2
        reply = netwk.send_message.call_args_list[0].args[0]
        assert isinstance(reply, messaging.InitReply)
        reply = netwk.send_message.call_args_list[1].args[0]
        assert isinstance(reply, messaging.Error)
        # Assert that the initial TrainingManager was set, but not the DP one.
        patch_tm.assert_called_once()
        patch_dp.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_with_dpsgd_error_setup(self) -> None:
        """Test error catching for DP-SGD setup with client-side failure."""
        if not DP_AVAILABLE:
            pytest.skip(reason="Unavailable DP features (missing Opacus).")
        # Set up a mock network receiving an InitRequest and a PrivacyRequest.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        msg_init = self._setup_mock_init_request(secagg=None, dpsgd=True)
        msg_priv, _ = self._setup_dpsgd_setup_query()
        netwk.recv_message.side_effect = [msg_init, msg_priv]
        # Set up a client wrapping the former network endpoint.
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Attempt running initialization, patching TrainingManager and
        # having DPTrainingManager fail.
        with patch_class_constructor(DPTrainingManager) as patch_dp:
            with patch_class_constructor(TrainingManager) as patch_tm:
                patch_dp.side_effect = TypeError
                with pytest.raises(RuntimeError):
                    await client.initialize()
        # Assert that TrainingManager was instantiated and DP one was called.
        patch_tm.assert_called_once()
        patch_dp.assert_called_once()
        # Assert that both messages were fetched, and an error was sent
        # after the DP-SGD setup failed.
        assert netwk.recv_message.call_count == 2
        assert netwk.send_message.call_count == 2
        reply = netwk.send_message.call_args_list[0].args[0]
        assert isinstance(reply, messaging.InitReply)
        reply = netwk.send_message.call_args_list[1].args[0]
        assert isinstance(reply, messaging.Error)

    def _setup_fairness_setup_query(
        self,
    ) -> Tuple[
        messaging.SerializedMessage[messaging.FairnessSetupQuery],
        messaging.FairnessSetupQuery,
    ]:
        """Setup a mock FairnessSetupQuery and a wrapping SerializedMessage."""
        fs_query = mock.create_autospec(
            messaging.FairnessSetupQuery, instance=True
        )
        msg_fair = mock.create_autospec(
            messaging.SerializedMessage, instance=True
        )
        msg_fair.message_cls = messaging.FairnessSetupQuery
        msg_fair.deserialize.return_value = fs_query
        return msg_fair, fs_query

    @pytest.mark.asyncio
    async def test_initialize_with_fairness(self) -> None:
        """Test that initialization with fairness works properly."""
        # Set up a mock network receiving an InitRequest,
        # then a FairnessSetupQuery.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        msg_init = self._setup_mock_init_request(secagg=None, fairness=True)
        msg_fair, fs_query = self._setup_fairness_setup_query()
        netwk.recv_message.side_effect = [msg_init, msg_fair]
        # Set up a client wrapping the former network endpoint.
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Attempt running initialization, patching fairness controller setup.
        with mock.patch.object(
            FairnessControllerClient, "from_setup_query"
        ) as patch_fcc:
            mock_controller = patch_fcc.return_value
            mock_controller.setup_fairness = mock.AsyncMock()
            await client.initialize()
        # Assert that a controller was instantiated and set up.
        patch_fcc.assert_called_once_with(
            query=fs_query, manager=client.trainmanager
        )
        mock_controller.setup_fairness.assert_awaited_once_with(
            netwk=client.netwk, secagg=None
        )
        assert client.fairness is mock_controller
        # Assert that a single InitReply was then sent to the server.
        netwk.send_message.assert_called_once()
        reply = netwk.send_message.call_args[0][0]
        assert isinstance(reply, messaging.InitReply)

    @pytest.mark.asyncio
    async def test_initialize_with_fairness_and_secagg(self) -> None:
        """Test that initialization with fairness and secagg works properly."""
        # Set up a mock network receiving an InitRequest with SecAgg,
        # then a SecaggSetupQuery and finally a FairnessSetupQuery.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        msg_init = self._setup_mock_init_request(
            secagg="mock-secagg", fairness=True
        )
        msg_sqry = mock.create_autospec(
            messaging.SerializedMessage, instance=True
        )
        msg_sqry.message_cls = SecaggSetupQuery
        msg_sqry.deserialize.return_value = mock.create_autospec(
            SecaggSetupQuery, instance=True
        )
        msg_fair, fs_query = self._setup_fairness_setup_query()
        netwk.recv_message.side_effect = [msg_init, msg_sqry, msg_fair]
        # Set up a client with that endpoint and a matching mock secagg.
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        secagg.secagg_type = "mock-secagg"
        client = FederatedClient(
            netwk=netwk, train_data=MOCK_DATASET, secagg=secagg
        )
        # Attempt running initialization, patching fairness controller setup.
        with mock.patch.object(
            FairnessControllerClient, "from_setup_query"
        ) as patch_fcc:
            mock_controller = patch_fcc.return_value
            mock_controller.setup_fairness = mock.AsyncMock()
            await client.initialize()
        # Assert that all three messages were fetched.
        assert netwk.recv_message.call_count == 3
        # Assert that a secagg controller was set up.
        secagg.setup_encrypter.assert_awaited_once_with(netwk, msg_sqry)
        # Assert that a fairness controller was instantiated
        # and then set up using the secagg controller.
        patch_fcc.assert_called_once_with(
            query=fs_query, manager=client.trainmanager
        )
        mock_controller.setup_fairness.assert_awaited_once_with(
            netwk=client.netwk, secagg=secagg.setup_encrypter.return_value
        )
        assert client.fairness is mock_controller
        # Assert that a single InitReply was then sent to the server.
        netwk.send_message.assert_called_once()
        reply = netwk.send_message.call_args[0][0]
        assert isinstance(reply, messaging.InitReply)

    @pytest.mark.asyncio
    async def test_initialize_with_fairness_error_wrong_message(self) -> None:
        """Test error catching for fairness setup with wrong second message."""
        # Set up a mock network receiving an InitRequest but wrong follow-up.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        msg_init = self._setup_mock_init_request(secagg=None, fairness=True)
        netwk.recv_message.return_value = msg_init
        # Set up a client wrapping the former network endpoint.
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Attempt running initialization, monitoring fairness controller setup.
        with mock.patch.object(
            FairnessControllerClient, "from_setup_query"
        ) as patch_fcc:
            with pytest.raises(RuntimeError):
                await client.initialize()
        # Assert that two messages were fetched, the first one answere with
        # an InitReply, the second with an Error.
        assert netwk.recv_message.call_count == 2
        assert netwk.send_message.call_count == 2
        reply = netwk.send_message.call_args_list[0].args[0]
        assert isinstance(reply, messaging.InitReply)
        reply = netwk.send_message.call_args_list[1].args[0]
        assert isinstance(reply, messaging.Error)
        # Assert that no fairness controller was set.
        patch_fcc.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_with_fairness_error_setup(self) -> None:
        """Test error catching for fairness setup with client-side failure."""
        # Set up a mock network receiving an InitRequest,
        # then a FairnessSetupQuery.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        msg_init = self._setup_mock_init_request(secagg=None, fairness=True)
        msg_fair, _ = self._setup_fairness_setup_query()
        netwk.recv_message.side_effect = [msg_init, msg_fair]
        # Set up a client wrapping the former network endpoint.
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Attempt running initialization, monitoring and forcing setup failure.
        with mock.patch.object(
            FairnessControllerClient, "from_setup_query"
        ) as patch_fcc:
            patch_fcc.side_effect = TypeError
            with pytest.raises(RuntimeError):
                await client.initialize()
        # Assert that setup was called (hence causing the exception).
        patch_fcc.assert_called_once()
        assert client.fairness is None
        # Assert that both messages were fetched, and an error was sent
        # after the fairness setup failed.
        assert netwk.recv_message.call_count == 2
        assert netwk.send_message.call_count == 2
        reply = netwk.send_message.call_args_list[0].args[0]
        assert isinstance(reply, messaging.InitReply)
        reply = netwk.send_message.call_args_list[1].args[0]
        assert isinstance(reply, messaging.Error)


class TestFederatedClientSetupSecagg:
    """Unit tests for 'FederatedClient.setup_secagg'."""

    @pytest.mark.asyncio
    async def test_setup_secagg(self) -> None:
        """Test that 'setup_secagg' sets up an encrypter."""
        # Set up a client with mock SecAgg controller.
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        client = FederatedClient(
            netwk=MOCK_NETWK, train_data=MOCK_DATASET, secagg=secagg
        )
        # Set up a mock serialized SecaggSetupQuery.
        msg = mock.create_autospec(messaging.SerializedMessage, instance=True)
        msg.message_cls = SecaggSetupQuery
        # Run the routine and verify that the setup protocol was triggered.
        await client.setup_secagg(msg)
        secagg.setup_encrypter.assert_called_once_with(client.netwk, query=msg)

    @pytest.mark.asyncio
    async def test_setup_secagg_no_secagg(self) -> None:
        """Test that 'setup_secagg' fails if no SecAgg is configured."""
        # Set up a client with mock NetworkClient.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Set up a mock serialized SecaggSetupQuery.
        msg = mock.create_autospec(messaging.SerializedMessage, instance=True)
        msg.message_cls = SecaggSetupQuery
        # Run the routine and verify that an Error was sent to the server.
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
        secagg.setup_encrypter.side_effect = ValueError
        client = FederatedClient(
            netwk=netwk, train_data=MOCK_DATASET, secagg=secagg
        )
        # Set up a mock serialized SecaggSetupQuery.
        msg = mock.create_autospec(messaging.SerializedMessage, instance=True)
        msg.message_cls = SecaggSetupQuery
        # Run the routine and verify that the exception was caught.
        # No message was sent because this is left up to the mocked subroutine.
        await client.setup_secagg(msg)
        secagg.setup_encrypter.assert_awaited_once_with(netwk=netwk, query=msg)
        netwk.send_message.assert_not_called()


class TestFederatedClientTrainingRound:
    """Unit tests for 'FederatedClient.training_round'."""

    @staticmethod
    def _finalize_mock_train_manager(
        train_manager: mock.Mock,
    ) -> mock.Mock:
        """Tweak a mock TrainingManager to enable using 'training_round'."""
        train_manager.model = mock.create_autospec(Model, instance=True)
        train_manager.training_round.return_value = mock.create_autospec(
            messaging.TrainReply
        )
        return train_manager

    @pytest.mark.asyncio
    async def test_training_round(self) -> None:
        """Test 'training_round' without SecAgg."""
        # Set up a client with a mock NetworkClient and mock TrainingManager.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        client = FederatedClient(netwk, train_data=MOCK_DATASET)
        train_manager = mock.create_autospec(TrainingManager, instance=True)
        client.trainmanager = self._finalize_mock_train_manager(train_manager)
        # Call the 'training_round' routine and verify expected actions.
        request = messaging.TrainRequest(
            round_i=1, weights=None, aux_var={}, batches={"batch_size": 32}
        )
        await client.training_round(request)
        train_manager.training_round.assert_called_once_with(request)
        netwk.send_message.assert_called_once_with(
            train_manager.training_round.return_value
        )

    @pytest.mark.asyncio
    async def test_training_round_secagg(self) -> None:
        """Test 'training_round' with SecAgg."""
        # Set up a client with a mock NetworkClient and SecaggConfigClient.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        client = FederatedClient(netwk, train_data=MOCK_DATASET, secagg=secagg)
        # Add a mock TrainingManager to it.
        train_manager = mock.create_autospec(TrainingManager, instance=True)
        client.trainmanager = self._finalize_mock_train_manager(train_manager)
        # Call the SecAgg setup routine.
        await client.setup_secagg(
            mock.create_autospec(messaging.SerializedMessage)
        )
        # Call the 'training_round' routine and verify expected actions.
        request = messaging.TrainRequest(
            round_i=1, weights=None, aux_var={}, batches={"batch_size": 32}
        )
        with mock.patch.object(
            secagg_messaging.SecaggTrainReply, "from_cleartext_message"
        ) as patched:
            await client.training_round(request)
        train_manager.training_round.assert_called_once_with(request)
        patched.assert_called_once_with(
            cleartext=train_manager.training_round.return_value,
            encrypter=secagg.setup_encrypter.return_value,
        )
        netwk.send_message.assert_called_once_with(patched.return_value)

    @pytest.mark.asyncio
    async def test_training_round_secagg_not_setup(self) -> None:
        """Test 'training_round' error with configured, not-setup SecAgg."""
        # Set up a client with a mock NetworkClient and SecaggConfigClient.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        client = FederatedClient(netwk, train_data=MOCK_DATASET, secagg=secagg)
        # Add a mock TrainingManager to it.
        train_manager = mock.create_autospec(TrainingManager, instance=True)
        client.trainmanager = self._finalize_mock_train_manager(train_manager)
        # Run the routine and verify that an Error message was sent.
        request = messaging.TrainRequest(
            round_i=1, weights=None, aux_var={}, batches={"batch_size": 32}
        )
        await client.training_round(request)
        netwk.send_message.assert_called_once()
        reply = netwk.send_message.call_args.args[0]
        assert isinstance(reply, messaging.Error)
        train_manager.training_round.assert_not_called()


class TestFederatedClientEvaluationRound:
    """Unit tests for 'FederatedClient.evaluation_round'."""

    @staticmethod
    def _finalize_mock_train_manager(
        train_manager: mock.Mock,
    ) -> mock.Mock:
        """Tweak a mock TrainingManager to return an EvaluationReply."""
        metric = mock.create_autospec(MetricState, instance=True)
        reply = messaging.EvaluationReply(
            loss=0.42, n_steps=42, t_spent=4.2, metrics={"metric": metric}
        )
        train_manager.evaluation_round.return_value = reply
        return train_manager

    @staticmethod
    def _setup_evaluation_request() -> messaging.EvaluationRequest:
        """Return an arbitrary EvaluationRequest message."""
        return messaging.EvaluationRequest(
            round_i=1,
            weights=None,
            batches={"batch_size": 32},
            n_steps=None,
            timeout=None,
        )

    @pytest.mark.asyncio
    async def test_evaluation_round(self) -> None:
        """Test 'evaluation_round' without SecAgg nor metrics removal."""
        # Set up a client with a mock NetworkClient and mock TrainingManager.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        client = FederatedClient(netwk, train_data=MOCK_DATASET)
        train_manager = mock.create_autospec(TrainingManager, instance=True)
        client.trainmanager = self._finalize_mock_train_manager(train_manager)
        # Call the 'evaluation_round' routine and verify expected actions.
        request = self._setup_evaluation_request()
        await client.evaluation_round(request)
        train_manager.evaluation_round.assert_called_once_with(request)
        netwk.send_message.assert_called_once_with(
            train_manager.evaluation_round.return_value
        )

    @pytest.mark.asyncio
    async def test_evaluation_round_no_share_metrics(self) -> None:
        """Test 'evaluation_round' with 'share_metrics=False'."""
        # Set up a client with a mock NetworkClient and mock TrainingManager.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        client = FederatedClient(
            netwk, train_data=MOCK_DATASET, share_metrics=False
        )
        train_manager = mock.create_autospec(TrainingManager, instance=True)
        client.trainmanager = self._finalize_mock_train_manager(train_manager)
        # Call the 'evaluation_round' routine and verify expected actions.
        request = self._setup_evaluation_request()
        await client.evaluation_round(request)
        train_manager.evaluation_round.assert_called_once_with(request)
        netwk.send_message.assert_called_once_with(
            train_manager.evaluation_round.return_value
        )
        # Verify that the return value's metrics have been cleared.
        assert not train_manager.evaluation_round.return_value.metrics

    @pytest.mark.asyncio
    async def test_evaluation_round_secagg(self) -> None:
        """Test 'evaluation_round' with SecAgg."""
        # Set up a client with a mock NetworkClient and SecaggConfigClient.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        client = FederatedClient(netwk, train_data=MOCK_DATASET, secagg=secagg)
        # Add a mock TrainingManager suitable to run evaluation rounds.
        train_manager = mock.create_autospec(TrainingManager, instance=True)
        client.trainmanager = self._finalize_mock_train_manager(train_manager)
        # Call the SecAgg setup routine.
        await client.setup_secagg(
            mock.create_autospec(messaging.SerializedMessage)
        )
        # Call the 'evaluation_round' routine and verify expected actions.
        request = self._setup_evaluation_request()
        with mock.patch.object(
            secagg_messaging.SecaggEvaluationReply, "from_cleartext_message"
        ) as patched:
            await client.evaluation_round(request)
        train_manager.evaluation_round.assert_called_once_with(request)
        patched.assert_called_once_with(
            cleartext=train_manager.evaluation_round.return_value,
            encrypter=secagg.setup_encrypter.return_value,
        )
        netwk.send_message.assert_called_once_with(patched.return_value)

    @pytest.mark.asyncio
    async def test_evaluation_round_secagg_not_setup(self) -> None:
        """Test 'evaluation_round' error with configured, not-setup SecAgg."""
        # Set up a client with a mock NetworkClient and SecaggConfigClient.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        client = FederatedClient(netwk, train_data=MOCK_DATASET, secagg=secagg)
        # Add a mock TrainingManager suitable to run evaluation rounds.
        train_manager = mock.create_autospec(TrainingManager, instance=True)
        client.trainmanager = self._finalize_mock_train_manager(train_manager)
        # Run the routine and verify that an Error message was sent.
        request = self._setup_evaluation_request()
        await client.evaluation_round(request)
        netwk.send_message.assert_called_once()
        reply = netwk.send_message.call_args.args[0]
        assert isinstance(reply, messaging.Error)
        train_manager.evaluation_round.assert_not_called()


class TestFederatedClientFairnessRound:
    """Unit tests for 'FederatedClient.fairness_round'."""

    @pytest.mark.parametrize("ckpt", [True, False], ids=["ckpt", "nockpt"])
    @pytest.mark.asyncio
    async def test_fairness_round(
        self,
        ckpt: bool,
    ) -> None:
        """Test 'fairness_round' with fairness and without SecAgg."""
        # Set up a client with a mock NetworkClient, TrainingManager,
        # FairnessClientController and optional Checkpointer.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        client = FederatedClient(netwk, train_data=MOCK_DATASET)
        client.trainmanager = mock.create_autospec(
            TrainingManager, instance=True
        )
        fairness = mock.create_autospec(
            FairnessControllerClient, instance=True
        )
        client.fairness = fairness
        if ckpt:
            client.ckptr = mock.create_autospec(Checkpointer, instance=True)
        # Call the 'fairness_round' routine and verify expected actions.
        request = messaging.FairnessQuery(round_i=0)
        await client.fairness_round(request)
        fairness.run_fairness_round.assert_awaited_once_with(
            netwk=netwk, query=request, secagg=None
        )
        # Verify that when a checkpointer is set, it is used.
        if ckpt:
            client.ckptr.save_metrics.assert_called_once_with(  # type: ignore
                metrics=fairness.run_fairness_round.return_value,
                prefix="fairness_metrics",
                append=False,  # first round, hence file creation or overwrite
                timestamp="round_0",
            )

    @pytest.mark.asyncio
    async def test_fairness_round_secagg(self) -> None:
        """Test 'fairness_round' with fairness and with SecAgg."""
        # Set up a client with a mock NetworkClient, TrainingManager,
        # FairnessClientController and SecaggConfigClient.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        client = FederatedClient(netwk, train_data=MOCK_DATASET, secagg=secagg)
        client.trainmanager = mock.create_autospec(
            TrainingManager, instance=True
        )
        fairness = mock.create_autospec(
            FairnessControllerClient, instance=True
        )
        client.fairness = fairness
        # Call the SecAgg setup routine.
        await client.setup_secagg(
            mock.create_autospec(messaging.SerializedMessage)
        )
        # Call the 'fairness_round' routine and verify expected actions.
        request = messaging.FairnessQuery(round_i=1)
        await client.fairness_round(request)
        fairness.run_fairness_round.assert_awaited_once_with(
            netwk=netwk,
            query=request,
            secagg=secagg.setup_encrypter.return_value,
        )

    @pytest.mark.asyncio
    async def test_fairness_round_fairness_not_setup(self) -> None:
        """Test 'fairness_round' without a fairness controller."""
        # Set up a client with a mock NetworkClient and TrainingManager,
        # but no fairness controller.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        client = FederatedClient(netwk, train_data=MOCK_DATASET)
        client.trainmanager = mock.create_autospec(
            TrainingManager, instance=True
        )
        # Verify that running the routine raises a RuntimeError.
        with pytest.raises(RuntimeError):
            await client.fairness_round(messaging.FairnessQuery(round_i=1))
        # Verify that an Error message was sent.
        netwk.send_message.assert_called_once()
        reply = netwk.send_message.call_args.args[0]
        assert isinstance(reply, messaging.Error)

    @pytest.mark.asyncio
    async def test_fairness_round_secagg_not_setup(self) -> None:
        """Test 'fairness_round' error with configured, not-setup SecAgg."""
        # Set up a client with a mock NetworkClient, TrainingManager,
        # FairnessClientController and SecaggConfigClient.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        secagg = mock.create_autospec(SecaggConfigClient, instance=True)
        client = FederatedClient(netwk, train_data=MOCK_DATASET, secagg=secagg)
        client.trainmanager = mock.create_autospec(
            TrainingManager, instance=True
        )
        fairness = mock.create_autospec(
            FairnessControllerClient, instance=True
        )
        client.fairness = fairness
        # Run the routine and verify that an Error message was sent.
        request = messaging.FairnessQuery(round_i=1)
        await client.fairness_round(request)
        netwk.send_message.assert_called_once()
        reply = netwk.send_message.call_args.args[0]
        assert isinstance(reply, messaging.Error)
        fairness.run_fairness_round.assert_not_called()


class TestFederatedClientMisc:
    """Unit tests for miscellaneous 'FederatedClient' methods."""

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
    async def test_cancel_training(self) -> None:
        """Test that a CancelTraining message is properly handled."""
        # Setup a client with a mock network client.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Have it process a CancelTraining message.
        message: messaging.SerializedMessage[messaging.CancelTraining] = (
            messaging.SerializedMessage.from_message_string(
                messaging.CancelTraining(reason="mock-reason").to_string()
            )
        )
        with pytest.raises(RuntimeError, match=".*mock-reason"):
            await client.handle_message(message)

    @pytest.mark.asyncio
    async def test_handle_message_error(self) -> None:
        """Test that 'handle_message' raises a ValueError on invalid input."""
        # Setup a client with a mock network client.
        netwk = mock.create_autospec(NetworkClient, instance=True)
        netwk.name = "client"
        client = FederatedClient(netwk=netwk, train_data=MOCK_DATASET)
        # Have it process an Error message.
        message: messaging.SerializedMessage[messaging.Error] = (
            messaging.SerializedMessage.from_message_string(
                messaging.Error(message="error-message").to_string()
            )
        )
        with pytest.raises(ValueError):
            await client.handle_message(message)
