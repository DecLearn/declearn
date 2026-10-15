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

"""Client-side main Federated Learning orchestrating class."""

import asyncio
import dataclasses
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from declearn import messaging
from declearn.communication.api import NetworkClient
from declearn.communication.utils import (
    NetworkClientConfig,
    verify_server_message_validity,
)
from declearn.dataset import Dataset
from declearn.fairness.api import FairnessControllerClient
from declearn.main.utils import Checkpointer, validate_optional_dataset_list
from declearn.messaging import Message, SerializedMessage
from declearn.secagg import messaging as secagg_messaging
from declearn.secagg import parse_secagg_config_client
from declearn.secagg.api import Encrypter, SecaggConfigClient, SecaggSetupQuery
from declearn.training import TrainingManager

__all__ = [
    "FederatedClient",
]


class FederatedClient:
    """Client-side Federated Learning orchestrating class.

    Notes
    -----
    You can access and configure the logger of each instance of this class
    using
    `logger = logging.getLogger("declearn.client-MY_CLIENT_NAME")`, and then
    adjust it as needed (e.g. `logger.setLevel(...)`).
    """

    # one-too-many attribute; pylint: disable=too-many-instance-attributes
    # pylint: disable-next=too-many-positional-arguments
    # TODO for 2.10 : remove deprecated "logger" argument
    def __init__(  # noqa: PLR0913
        self,
        netwk: Union[NetworkClient, NetworkClientConfig, Dict[str, Any], str],
        train_data: List[Dataset],
        valid_data: Optional[List[Dataset]] = None,
        checkpoint: Union[Checkpointer, Dict[str, Any], str, None] = None,
        secagg: Union[SecaggConfigClient, Dict[str, Any], None] = None,
        share_metrics: bool = True,
        logger: Union[logging.Logger, str, None] = None,
        verbose: bool = True,
    ) -> None:
        """Instantiate a client to participate in a federated learning task.

        Parameters
        ----------
        netwk: NetworkClient or NetworkClientConfig or dict or str
            NetworkClient communication endpoint instance, or configuration
            dict, dataclass or path to a TOML file enabling its instantiation.
            In the latter three cases, the object's default logger will be set
            to that of this `FederatedClient`.
        train_data: Dataset or str
            Dataset instance wrapping the training data.
        valid_data: Dataset or str or None
            Optional Dataset instance wrapping validation data.
            If None, run evaluation rounds over `train_data`.
        checkpoint: Checkpointer or dict or str or None, default=None
            Optional Checkpointer instance or instantiation dict to be
            used so as to save round-wise model, optimizer and metrics.
            If a single string is provided, treat it as the checkpoint
            folder path and use default values for other parameters.
        secagg: SecaggConfigClient or dict or None, default=None
            Optional SecAgg config and setup controller
            or dict of kwargs to set one up.
        share_metrics: bool, default=True
            Whether to share evaluation metrics with the server,
            or save them locally and only send the model's loss.
            This may prevent information leakage, e.g. as to the
            local distribution of target labels or values.
        logger: logging.Logger or str or None, default=None,
            Deprecated in v2.8, removed in v2.10.
            Not used anymore.
        verbose: bool, default=True
            Whether to verbose about ongoing operations.
            If True, display progress bars during training and validation
            rounds.
        """
        if logger is not None:
            warnings.warn(
                "Argument 'logger' is deprecated and useless now, it will be "
                "removed in 2.10. "
                "To customize the instance logger, you may use instead "
                "logging utils from `declearn.utils` or the 'logging' Python "
                "module.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.netwk = self._parse_netwk(netwk)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(f"declearn.client-{self.netwk.name}")

        # Assign the wrapped training dataset.
        self.train_data = validate_optional_dataset_list(
            train_data, "train_data"
        )
        self.valid_data = validate_optional_dataset_list(
            valid_data, "valid_data", len(self.train_data)
        )
        # Assign an optional checkpointer.
        if checkpoint is not None:
            checkpoint = Checkpointer.from_specs(checkpoint)
        self.ckptr = checkpoint
        # Assign the optional SecAgg config and declare an Encrypter slot.
        self.secagg = self._parse_secagg(secagg)
        self._encrypter: Optional[Encrypter] = None
        # Record the metric-sharing and verbosity bool values.
        self.share_metrics = bool(share_metrics)
        if (self.secagg is not None) and not self.share_metrics:
            msg = (
                "Disabling metrics' sharing with SecAgg enabled is likely"
                "to cause errors, unless each and every client does so."
            )
            self.logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=-1)
        self.verbose = bool(verbose)

    @staticmethod
    def _parse_netwk(netwk) -> NetworkClient:
        """Parse 'netwk' instantiation argument and return 'NetworkClient'
        instance.
        """
        # Case when a NetworkClient instance is provided: return.
        if isinstance(netwk, NetworkClient):
            return netwk
        # Case when a NetworkClientConfig is expected: verify or parse.
        if isinstance(netwk, NetworkClientConfig):
            config = netwk
        elif isinstance(netwk, str):
            config = NetworkClientConfig.from_toml(netwk)
        elif isinstance(netwk, dict):
            config = NetworkClientConfig.from_params(**netwk)
        else:
            raise TypeError(
                "'netwk' should be a 'NetworkClient' instance or the valid "
                f"configuration of one, not '{type(netwk)}'"
            )
        # Instantiate from the (parsed) config.
        return config.build_client()

    @staticmethod
    def _parse_secagg(
        secagg: Union[SecaggConfigClient, Dict[str, Any], None],
    ) -> Optional[SecaggConfigClient]:
        """Parse 'secagg' instantiation argument."""
        if secagg is None:
            return None
        if isinstance(secagg, SecaggConfigClient):
            return secagg
        if isinstance(secagg, dict):
            try:
                return parse_secagg_config_client(**secagg)
            except Exception as exc:
                raise TypeError("Failed to parse 'secagg' inputs.") from exc
        raise TypeError(
            "'secagg' should be a 'SecaggConfigClient' instance or a dict "
            f"of keyword arguments to set one up, not '{type(secagg)}'."
        )

    def run(
        self,
    ) -> None:
        """Participate in the federated learning process.

        * Connect to the orchestrating `FederatedServer` and register
          for training, sharing some metadata about `self.train_data`.
        * Await initialization instructions to spawn the Model that is
          to be trained and the local Optimizer used to do so.
        * Participate in training and evaluation rounds based on the
          server's requests, checkpointing the model and local loss.
        * Expect instructions to stop training, or to cancel it in
          case errors are reported during the process.
        """
        asyncio.run(self.async_run())

    async def async_run(
        self,
    ) -> None:
        """Participate in the federated learning process.

        Note: this method is the async backend of `self.run`.
        """
        async with self.netwk:
            # Register for training, then collect initialization information.
            await self.register()
            await self.initialize()

            # Process server instructions as they come.
            while True:
                ser_msg: SerializedMessage = await self.netwk.recv_message()
                stoprun = await self.handle_message(ser_msg)
                if stoprun:
                    break

    async def handle_message(
        self,
        message: SerializedMessage,
    ) -> bool:
        """Handles an incoming message from the server. The Inference round is
        by default always handled post training upon an incoming request
        from the server.

        Parameters
        ----------
        message: SerializedMessage
            Serialized message requiring triage and processing.

        Returns
        -------
        exit_loop: bool
            Whether to interrupt the client's message-receiving loop.
        """
        exit_loop = False
        if issubclass(message.message_cls, messaging.MultiModelTrainRequest):
            # train model1, model2
            await self.training_round(message.deserialize())
        elif issubclass(
            message.message_cls, messaging.MultiModelEvaluationRequest
        ):
            await self.evaluation_round(message.deserialize())
        elif issubclass(
            message.message_cls, messaging.MultiModelFairnessQuery
        ):
            await self.fairness_round(message.deserialize())
        elif issubclass(message.message_cls, SecaggSetupQuery):
            await self.setup_secagg(message)  # note: keep serialized
        elif issubclass(message.message_cls, messaging.StopTraining):
            await self.stop_training(message.deserialize())
            exit_loop = True

        elif issubclass(message.message_cls, messaging.CancelTraining):
            await self.cancel_training(message.deserialize())
        else:
            error = "Unexpected message type received from server: "
            error += message.message_cls.__name__
            self.logger.error(error)
            raise ValueError(error)
        return exit_loop

    async def register(
        self,
    ) -> None:
        """Register for participation in the federated learning process.

        Raises
        ------
        RuntimeError
            If registration has failed 10 times (with a 1 minute delay
            between connection and registration attempts).
        """
        for i in range(10):  # max_attempts (10)
            self.logger.info(
                "Attempting to join training (attempt n°%s)", i + 1
            )
            registered = await self.netwk.register()
            if registered:
                break
            await asyncio.sleep(60)  # delay_retries (1 minute)
        else:
            raise RuntimeError("Failed to register for training.")

    async def initialize(
        self,
    ) -> None:
        """Set up Models and Optimizers based on server instructions.

        Await server instructions (as an InitRequest message) and conduct
        initialization for all models.

        Raises
        ------
        RuntimeError
            If initialization failed, either because the message was not
            received or was of incorrect type, because the SecAgg
            configuration is inconsistent, or because instantiation of the
            training managers failed.
        """
        # Await initialization instructions.
        self.logger.info("Awaiting initialization instructions from server.")
        received: SerializedMessage = await self.netwk.recv_message()

        # If a MetadataQuery is received, process it, then await InitRequest.
        if issubclass(received.message_cls, messaging.MetadataQuery):
            await self._collect_and_send_metadata_of_models(
                received.deserialize()
            )
            received = await self.netwk.recv_message()

        # Ensure that an CausalInitRequest was received.
        message = await verify_server_message_validity(
            self.netwk,
            received,
            expected=messaging.InitRequest,
        )
        # Verify that SecAgg type is coherent across peers.
        secagg_type = None if self.secagg is None else self.secagg.secagg_type
        if message.secagg != secagg_type:
            error = (
                "SecAgg configuration mismatch: server set "
                f"'{message.secagg}', client set '{secagg_type}'."
            )
            self.logger.error(error)
            await self.netwk.send_message(messaging.Error(error))
            raise RuntimeError(f"Initialization failed: {error}.")

        # Perform initialization, catching errors to report them to the server.
        try:
            self.trainmanagers = []

            for idx in range(len(message.models)):
                self.trainmanagers.append(
                    TrainingManager(
                        model=message.models[idx],
                        optim=message.optimizers[idx],
                        aggrg=message.aggrg[idx],
                        train_data=self.train_data[idx],
                        valid_data=self.valid_data[idx],
                        metrics=message.metrics[idx],
                        logger=self.logger,
                        verbose=self.verbose,
                    )
                )

            self.fairness: List[Optional[FairnessControllerClient]] = [
                None
            ] * len(self.trainmanagers)
        except Exception as exc:
            await self.netwk.send_message(messaging.Error(repr(exc)))
            raise RuntimeError("Initialization failed.") from exc

        # Send back an empty message to indicate that things went fine.
        self.logger.info("Notifying the server that initialization went fine.")
        await self.netwk.send_message(messaging.InitReply())

        # If instructed to do so, run additional steps to set up DP-SGD.
        if message.dpsgd:
            await self._initialize_dpsgd()

        # Initialize SecAgg once, if required (it's the same across all models)
        if self.secagg is not None:
            secagg_received: SerializedMessage = (
                await self.netwk.recv_message()
            )
            await self.setup_secagg(secagg_received)

        # If instructed to do so, run additional steps to enforce fairness.
        # at least one model's fairness option is true to
        # start the fairness initialization
        if any(message.fairness):
            for model_index, fairness_enabled in enumerate(message.fairness):
                if fairness_enabled:
                    await self._initialize_fairness(model_index)

        # Optionally checkpoint the received models and optimizers.
        if self.ckptr:
            for idx, trainmanager in enumerate(self.trainmanagers):
                self.ckptr.set_model_folder(idx=idx)
                self.ckptr.checkpoint(
                    model=trainmanager.model,
                    optimizer=trainmanager.optim,
                    first_call=True,
                )

    async def _collect_and_send_metadata_of_models(
        self,
        message: messaging.MetadataQuery,
    ) -> None:
        """Collect and report metadata for all training datasets."""

        metadata_list = [
            dataclasses.asdict(data.get_data_specs())
            for data in self.train_data
        ]
        _metadata: Dict = {}
        for model_index, metadata in enumerate(metadata_list):
            if missing := set(message.fields[f"{model_index}"]).difference(
                metadata
            ):
                err_msg = (
                    f"Metadata query for model {model_index} "
                    f"contains undefined fields: {missing}."
                )
                await self.netwk.send_message(messaging.Error(err_msg))
                raise RuntimeError(err_msg)

            data_info = {
                key: metadata[key] for key in message.fields[f"{model_index}"]
            }
            _metadata[f"{model_index}"] = data_info
        self.logger.info(
            "Sending models metadata to the server",
        )

        await self.netwk.send_message(
            messaging.MetadataReply(data_info=_metadata)
        )

    async def _initialize_dpsgd(
        self,
    ) -> None:
        """Set up differentially-private training as part of initialization.

        This method wraps the `make_private` one in the context of
        `initialize` and should never be called in another context.
        """
        received = await self.netwk.recv_message()
        try:
            message = await verify_server_message_validity(
                self.netwk, received, expected=messaging.PrivacyRequest
            )
        except Exception as exc:
            raise RuntimeError("DP-SGD initialization failed.") from exc
        self.logger.info("Received DP-SGD setup instructions.")
        try:
            self.make_private(message)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(
                "Exception encountered in `make_private`: %s",
                exc,
            )
            await self.netwk.send_message(messaging.Error(repr(exc)))
            raise RuntimeError("DP-SGD initialization failed.") from exc
        # If things went right, notify the server.
        self.logger.info("Notifying the server that DP-SGD setup went fine.")
        await self.netwk.send_message(messaging.PrivacyReply())

    def make_private(
        self,
        message: messaging.PrivacyRequest,
    ) -> None:
        """Set up differentially private training using DP-SGD.

        Replace each training manager with a DPTrainingManager and configure
        it for differentially private training.

        Note that this method imports `declearn.training.dp`, which may raise
        an error if the optional dependency `opacus` is unavailable.

        Parameters
        ----------
        message: messaging.PrivacyRequest
            Instructions from the server regarding the DP-SGD setup.

        Raises
        ------
        RuntimeError
            If initialization of a DP training manager fails.
        """
        # Lazy import to avoid loading optional dependencies unnecessarily.
        # pylint: disable=import-outside-toplevel
        try:
            from declearn.training.dp import DPTrainingManager  # noqa: PLC0415

            self.trainmanagers = [
                DPTrainingManager(
                    model=manager.model,
                    optim=manager.optim,
                    aggrg=manager.aggrg,
                    train_data=manager.train_data,
                    valid_data=manager.valid_data,
                    metrics=manager.metrics,
                    logger=self.logger,
                    verbose=manager.verbose,
                )
                for manager in self.trainmanagers
            ]

            for manager in self.trainmanagers:
                manager.make_private(message)

        except Exception as exc:
            raise RuntimeError(
                "Initialization of private training failed."
            ) from exc

    async def _initialize_fairness(self, model_index: int) -> None:
        """Set up fairness enforcement for a specific model."""

        try:
            # Receive the setup query for this model.
            received = await self.netwk.recv_message()
            query = await verify_server_message_validity(
                self.netwk,
                received,
                expected=messaging.FairnessSetupQuery,
            )

        except Exception as exc:
            error = f"Fairness initialization failed: {repr(exc)}."
            self.logger.critical(error)
            raise RuntimeError(error) from exc

        self.logger.info(
            "Received fairness setup instructions for model %s.",
            model_index,
        )

        try:
            trainmanager = self.trainmanagers[model_index]

            fairness_controller = FairnessControllerClient.from_setup_query(
                query=query,
                manager=trainmanager,
            )

            await fairness_controller.setup_fairness(
                netwk=self.netwk,
                secagg=self._encrypter,
            )

            # Store the controller at the corresponding model index.
            self.fairness[model_index] = fairness_controller

        except Exception as exc:
            error = (
                f"Fairness-aware federated learning setup failed: {repr(exc)}."
            )
            self.logger.critical(error)
            await self.netwk.send_message(messaging.Error(error))
            raise RuntimeError(error) from exc

    async def setup_secagg(
        self,
        received: SerializedMessage[SecaggSetupQuery],
    ) -> None:
        """Participate in a SecAgg setup protocol.

        Process a setup request from the server, run a method-specific
        protocol (that may involve additional communications) and update
        the held SecAgg `Encrypter` with the resulting one.

        Parameters
        ----------
        received:
            Serialized `SecaggSetupQuery` request received from the server,
            the exact type of which depends on the SecAgg method being set.
        """
        # If no SecAgg setup controller was set, send an Error message.
        if self.secagg is None:
            error = (
                "Received a SecAgg setup request, but SecAgg is not "
                "configured to be used."
            )
            self.logger.error(error)
            await self.netwk.send_message(messaging.Error(error))
            return
        # Otherwise, participate in the SecAgg setup protocol.
        self.logger.info("Received a SecAgg setup request.")
        try:
            self._encrypter = await self.secagg.setup_encrypter(
                netwk=self.netwk, query=received
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            self.logger.error("SecAgg setup failed: %s", repr(exc))

    async def training_round(
        self,
        message: messaging.MultiModelTrainRequest,
    ) -> None:
        """Run a local training round.

        If an exception is raised during the local process, wrap
        it as an Error message and send it to the server instead
        of raising it.

        Parameters
        ----------
        message: CausalTrainRequest
            Instructions from the server regarding the training round.
        """
        replies = []
        self.logger.info("Training the models")

        for idx, msg in enumerate(message.requests):
            try:
                self.logger.info(f"Model {idx + 1}")
                reply = await self.training_step(msg, idx)
                replies.append(reply)

            except Exception as exc:
                self.logger.exception(
                    "Training failed for model %d in round %d",
                    idx + 1,
                    msg.round_i,
                )
                await self.netwk.send_message(messaging.Error(str(exc)))
                return

        if self._encrypter is not None:
            replies = secagg_messaging.MultiModelSecaggTrainReply(
                replies=replies
            )
        else:
            replies = messaging.MultiModelTrainReply(replies=replies)

        # Send training results (or error message) to the server.
        await self.netwk.send_message(replies)

    async def training_step(
        self, message: messaging.TrainRequest, model_index: int = 0
    ):
        """Run one local training step for a specific model.

        Verify SecAgg setup, execute the training round, collect and optionally
        checkpoint training losses, and encrypt the reply if SecAgg is enabled.

        Parameters
        ----------
        message: messaging.TrainRequest
            Instructions from the server for the training round.
        model_index: int, default=0
            Index of the model to train.

        Returns
        -------
        messaging.TrainReply or secagg_messaging.SecaggTrainReply
            The training results, optionally encrypted for secure aggregation.

        Raises
        ------
        RuntimeError
            If SecAgg is configured but has not been set up.
        """
        # at this level update the checkpoint to point to the current model
        # When SecAgg is to be used, verify that it was set up.
        if self.secagg is not None and self._encrypter is None:
            error = (
                f"Refusing to participate in training round {message.round_i} "
                "as SecAgg is configured to be used but was not set up."
            )
            self.logger.error(error)
            raise RuntimeError(error)

        trainmanager = self.trainmanagers[model_index]

        # Run the training round.

        reply: Message = trainmanager.training_round(message)
        # Collect and optionally record batch-wise training losses.
        # Note: collection enables purging them from memory.
        losses = trainmanager.model.collect_training_losses()

        if self.ckptr is not None:
            self.ckptr.set_model_folder(model_index)

            self.ckptr.save_metrics(
                metrics={"training_losses": np.array(losses)},
                prefix="training_losses",
                append=True,
                timestamp=f"round_{message.round_i}",
            )

        # Optionally SecAgg-encrypt the reply.
        if self._encrypter is not None and isinstance(
            reply, messaging.TrainReply
        ):
            reply = secagg_messaging.SecaggTrainReply.from_cleartext_message(
                cleartext=reply,
                encrypter=self._encrypter,
            )
        return reply

    async def evaluation_round(
        self,
        message: messaging.MultiModelEvaluationRequest,
    ) -> None:
        """Run a local evaluation round.

        If an exception is raised during the local process, wrap
        it as an Error message and send it to the server instead
        of raising it.

        If a checkpointer is set, record the local loss, and the
        model weights received from the server.

        Parameters
        ----------
        message: CausalEvaluationRequest
            Instructions from the server regarding the evaluation round.
        """
        replies = []
        try:
            for idx, msg in enumerate(message.requests):
                reply = await self.evaluation_step(msg, idx)
                replies.append(reply)

        except Exception as exc:
            self.logger.exception(
                "Evaluation failed",
            )
            await self.netwk.send_message(messaging.Error(str(exc)))
            return

        reply = messaging.MultiModelEvaluationReply(replies)
        # override the responses with the correct type
        # we assume that all the replies are either all
        # secureagged or not.

        if self._encrypter is not None:
            reply = secagg_messaging.MultiModelSecaggEvaluationReply(replies)

        await self.netwk.send_message(reply)

    async def evaluation_step(
        self, message: messaging.EvaluationRequest, model_index: int = 0
    ) -> messaging.EvaluationReply | messaging.Error:
        """Evaluate a specific model for a given evaluation round.

        Verify SecAgg setup, run local evaluation, optionally checkpoint the
        results, filter shared metrics, and encrypt the reply if SecAgg is
        enabled.

        Parameters
        ----------
        message: messaging.EvaluationRequest
            Instructions from the server for the evaluation round.
        model_index: int, default=0
            Index of the model to evaluate.

        Returns
        -------
        messaging.EvaluationReply | messaging.Error
            The evaluation results or an error message.

        Raises
        ------
        RuntimeError
            If SecAgg is configured but has not been set up.
        """
        # When SecAgg is to be used, verify that it was set up.
        if self.secagg is not None and self._encrypter is None:
            error = (
                "Refusing to participate in evaluation round "
                f"{message.round_i} as SecAgg is configured to be used "
                "but was not set up."
            )
            self.logger.error(error)
            raise RuntimeError(error)

        trainmanager = self.trainmanagers[model_index]

        # Run the evaluation round.
        reply: Message = trainmanager.evaluation_round(message)
        # Post-process the results.
        if isinstance(reply, messaging.EvaluationReply):
            # Optionally checkpoint the model, optimizer and local loss.
            if self.ckptr:
                self.ckptr.set_model_folder(model_index)
                self.ckptr.checkpoint(
                    model=trainmanager.model,
                    optimizer=trainmanager.optim,
                    metrics=trainmanager.metrics.get_result(),
                )

        # Optionally prevent sharing metrics (save for the loss).
        if not self.share_metrics:
            reply.metrics.clear()

        # Optionally SecAgg-encrypt results.
        if self._encrypter is not None:
            msg_cls = secagg_messaging.SecaggEvaluationReply
            reply = msg_cls.from_cleartext_message(
                cleartext=reply,
                encrypter=self._encrypter,
            )

        return reply

    async def fairness_round(
        self, queries: messaging.MultiModelFairnessQuery
    ) -> None:
        """Handle a server request to run a fairness-related round.

        The nature of the round depends on the fairness-aware learning
        algorithm that was optionally set up during the initialization
        phase. In case no such algorithm was set up, this method will
        raise a process-crashing exception.

        Parameters
        ----------
        query:
            `MultiModelFairnessQuery` message from the server.

        Raises
        ------
        RuntimeError
            If no fairness controller was set up for this instance.
        """

        try:
            for idx, query in enumerate(queries.requests):
                await self.fairness_step(query, idx)
        except Exception as exc:
            self.logger.exception(
                "Fairness round failed",
            )
            await self.netwk.send_message(messaging.Error(str(exc)))
            return

    async def fairness_step(
        self, query: messaging.FairnessQuery, model_index: int = 0
    ) -> None:
        """Handle a server request to run a fairness-related round."""

        # Retrieve and validate the fairness controller.
        fairness_controller = self.fairness[model_index]

        if fairness_controller is None:
            error = (
                "Received a query to participate in a fairness round, "
                "but no fairness controller was set up."
            )
            self.logger.critical(error)
            raise RuntimeError(error)

        # When SecAgg is to be used, verify that it was set up.
        if self.secagg is not None and self._encrypter is None:
            error = (
                "Refusing to participate in fairness-related round "
                f"{query.round_i} as SecAgg is configured to be used "
                "but was not set up."
            )
            self.logger.error(error)
            raise RuntimeError(error)

        # Run the controller's routine.
        metrics = await fairness_controller.run_fairness_round(
            netwk=self.netwk,
            query=query,
            secagg=self._encrypter,
        )

        # Optionally save computed fairness metrics.
        if self.ckptr is not None:
            self.ckptr.set_model_folder(model_index)
            self.ckptr.save_metrics(
                metrics=metrics,
                prefix="fairness_metrics",
                append=bool(query.round_i),
                timestamp=f"round_{query.round_i}",
            )

    async def stop_training(
        self,
        message: messaging.StopTraining,
    ) -> None:
        """Handle a server request to stop training.

        Parameters
        ----------
        message: StopTraining
            StopTraining message received from the server.
        """
        self.logger.info("Global Training is now over,")

        for key in message.weights.keys():
            idx = int(key)
            self.logger.info(
                f"Model {idx + 1} after "
                + f"{message.rounds} rounds"
                + f", Global loss: {message.loss[key]}"
            )
            self.logger.info(
                "Processing final weights for model %d.",
                idx + 1,
            )

            if self.ckptr:
                self.ckptr.set_model_folder(idx)
                path = os.path.join(
                    self.ckptr.folder,
                    "model_state_best.mpk",
                )
                self.logger.info(
                    "Checkpointing final weights under %s.",
                    path,
                )

                assert self.trainmanagers[idx] is not None
                self.trainmanagers[idx].model.set_weights(
                    message.weights[key],
                    trainable=True,
                )

                self.ckptr.save_model(
                    self.trainmanagers[idx].model,
                    timestamp="best",
                )

    async def cancel_training(
        self,
        message: messaging.CancelTraining,
    ) -> None:
        """Handle a server request to cancel training.

        Parameters
        ----------
        message: CancelTraining
            CancelTraining message received from the server.
        """
        error = "Training was cancelled by the server, with reason:\n"
        error += message.reason
        self.logger.warning(error)
        raise RuntimeError(error)
