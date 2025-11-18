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

"""Client-side main Federated Learning orchestrating class."""

import asyncio
import dataclasses
import logging
import os
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from declearn import messaging
from declearn.communication.api import NetworkClient
from declearn.communication.utils import (
    NetworkClientConfig,
    verify_server_message_validity,
)
from declearn.dataset import Dataset
from declearn.fairness.api import FairnessControllerClient
from declearn.main.utils import Checkpointer
from declearn.messaging import Message, SerializedMessage
from declearn.secagg import messaging as secagg_messaging
from declearn.secagg import parse_secagg_config_client
from declearn.secagg.api import Encrypter, SecaggConfigClient, SecaggSetupQuery
from declearn.training import TrainingManager
from declearn.utils import LOGGING_LEVEL_MAJOR, get_logger

__all__ = [
    "FederatedClient",
]


class FederatedClient:
    """Client-side Federated Learning orchestrating class."""

    # one-too-many attribute; pylint: disable=too-many-instance-attributes
    # pylint: disable-next=too-many-positional-arguments
    def __init__(  # noqa: PLR0913
        self,
        netwk: Union[NetworkClient, NetworkClientConfig, Dict[str, Any], str],
        train_data: Union[Dataset, str],
        valid_data: Optional[Union[Dataset, str]] = None,
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
            (DEPRECATED) May be a path to a JSON dump file.
        valid_data: Dataset or str or None
            Optional Dataset instance wrapping validation data.
            If None, run evaluation rounds over `train_data`.
            (DEPRECATED) May be a path to a JSON dump file.
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
            Logger to use, or name of a logger to set up with
            `declearn.utils.get_logger`.
            If None, use `type(self):netwk.name`.
        verbose: bool, default=True
            Whether to verbose about ongoing operations.
            If True, display progress bars during training and validation
            rounds. If False and `logger is None`, set the logger's level
            to filter off most routine information.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        # Assign the wrapped NetworkClient.
        self.netwk, replace_netwk_logger = self._parse_netwk(netwk)
        # Assign the logger and optionally replace that of the network client.
        if not isinstance(logger, logging.Logger):
            logger = get_logger(
                name=logger or f"{type(self).__name__}-{self.netwk.name}",
                level=logging.INFO if verbose else LOGGING_LEVEL_MAJOR,
            )
        self.logger = logger
        if replace_netwk_logger:
            self.netwk.logger = self.logger
        # Assign the wrapped training dataset.
        if not isinstance(train_data, Dataset):
            raise TypeError("'train_data' should be a Dataset.")
        self.train_data = train_data
        # Assign the wrapped validation dataset (if any).
        if not (valid_data is None or isinstance(valid_data, Dataset)):
            raise TypeError("'valid_data' should be a Dataset.")
        self.valid_data = valid_data
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
        # Create slots that are (opt.) populated during initialization.
        self.trainmanager: Optional[TrainingManager] = None
        self.fairness: Optional[FairnessControllerClient] = None

    @staticmethod
    def _parse_netwk(netwk) -> Tuple[NetworkClient, bool]:
        """Parse 'netwrk' instantiation argument.

        Return both a 'NetworkClient' instance and a bool indicating
        whether that instance's logger should be replaced with that
        of the client (set up at a latter step).
        """
        # Case when a NetworkClient instance is provided: return.
        if isinstance(netwk, NetworkClient):
            return netwk, False
        # Case when a NetworkClientConfig is expected: verify or parse.
        if isinstance(netwk, NetworkClientConfig):
            config = netwk
        elif isinstance(netwk, str):
            config = NetworkClientConfig.from_toml(netwk)
        elif isinstance(netwk, dict):
            replace_netwk_logger = netwk.get("logger", None) is None
            config = NetworkClientConfig.from_params(**netwk)
        else:
            raise TypeError(
                "'netwk' should be a 'NetworkClient' instance or the valid "
                f"configuration of one, not '{type(netwk)}'"
            )
        # Instantiate from the (parsed) config.
        replace_netwk_logger = config.logger is None
        return config.build_client(), replace_netwk_logger

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
                message = await self.netwk.recv_message()
                stoprun = await self.handle_message(message)
                if stoprun:
                    break

    async def handle_message(
        self,
        message: SerializedMessage,
    ) -> bool:
        """Handle an incoming message from the server.

        Parameters
        ----------
        message: SerializedMessage
            Serialized message that needs triage and processing.

        Returns
        -------
        exit_loop: bool
            Whether to interrupt the client's message-receiving loop.
        """
        exit_loop = False
        if issubclass(message.message_cls, messaging.TrainRequest):
            await self.training_round(message.deserialize())
        elif issubclass(message.message_cls, messaging.EvaluationRequest):
            await self.evaluation_round(message.deserialize())
        elif issubclass(message.message_cls, messaging.FairnessQuery):
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
        """Set up a Model and an Optimizer based on server instructions.

        Await server instructions (as an InitRequest message) and conduct
        initialization.

        Raises
        ------
        RuntimeError
            If initialization failed, either because the message was not
            received or was of incorrect type, or because instantiation
            of the objects it specifies failed.

        Returns
        -------
        model: Model
            Model that is to be trained (with shared initial parameters).
        optim: Optimizer
            Optimizer that is to be used locally to train the model.
        """
        # Await initialization instructions.
        self.logger.info("Awaiting initialization instructions from server.")
        received = await self.netwk.recv_message()
        # If a MetadataQuery is received, process it, then await InitRequest.
        if issubclass(received.message_cls, messaging.MetadataQuery):
            await self._collect_and_send_metadata(received.deserialize())
            received = await self.netwk.recv_message()
        # Ensure that an 'InitRequest' was received.
        message = await verify_server_message_validity(
            self.netwk, received, expected=messaging.InitRequest
        )
        # Verify that SecAgg type is coherent across peers.
        secagg_type = None if self.secagg is None else self.secagg.secagg_type
        if message.secagg != secagg_type:
            error = (
                "SecAgg configurgation mismatch: server set "
                f"'{message.secagg}', client set '{secagg_type}'."
            )
            self.logger.error(error)
            await self.netwk.send_message(messaging.Error(error))
            raise RuntimeError(f"Initialization failed: {error}.")
        # Perform initialization, catching errors to report them to the server.
        try:
            self.trainmanager = TrainingManager(
                model=message.model,
                optim=message.optim,
                aggrg=message.aggrg,
                train_data=self.train_data,
                valid_data=self.valid_data,
                metrics=message.metrics,
                logger=self.logger,
                verbose=self.verbose,
            )
        except Exception as exc:
            await self.netwk.send_message(messaging.Error(repr(exc)))
            raise RuntimeError("Initialization failed.") from exc
        # Send back an empty message to indicate that things went fine.
        self.logger.info("Notifying the server that initialization went fine.")
        await self.netwk.send_message(messaging.InitReply())
        # If instructed to do so, run additional steps to set up DP-SGD.
        if message.dpsgd:
            await self._initialize_dpsgd()
        # If instructed to do so, run additional steps to enforce fairness.
        if message.fairness:
            await self._initialize_fairness()
        # Optionally checkpoint the received model and optimizer.
        if self.ckptr:
            self.ckptr.checkpoint(
                model=self.trainmanager.model,
                optimizer=self.trainmanager.optim,
                first_call=True,
            )

    async def _collect_and_send_metadata(
        self,
        message: messaging.MetadataQuery,
    ) -> None:
        """Collect and report some metadata based on server instructions."""
        self.logger.info("Collecting metadata to send to the server.")
        metadata = dataclasses.asdict(self.train_data.get_data_specs())
        if missing := set(message.fields).difference(metadata):
            err_msg = f"Metadata query for undefined fields: {missing}."
            await self.netwk.send_message(messaging.Error(err_msg))
            raise RuntimeError(err_msg)
        data_info = {key: metadata[key] for key in message.fields}
        self.logger.info(
            "Sending training dataset metadata to the server: %s.",
            list(data_info),
        )
        await self.netwk.send_message(messaging.MetadataReply(data_info))

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
                "Exception encountered in `make_private`: %s", exc
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
        """Set up differentially-private training, using DP-SGD.

        Based on the server message, replace the wrapped `trainmanager`
        attribute by an instance of a subclass that provides with DP-SGD.

        Note that this method triggers the import of `declearn.main.privacy`
        which may result in an error if the third-party dependency 'opacus'
        is not available.

        Parameters:
        ----------
        message: PrivacyRequest
            Instructions from the server regarding the DP-SGD setup.
        """
        assert self.trainmanager is not None
        # fmt: off
        # lazy-import the DPTrainingManager, that involves some optional,
        # heavy-loadtime dependencies; pylint: disable=import-outside-toplevel
        from declearn.training.dp import DPTrainingManager # noqa: I001, PLC0415

        # pylint: enable=import-outside-toplevel
        self.trainmanager = DPTrainingManager(
            model=self.trainmanager.model,
            optim=self.trainmanager.optim,
            aggrg=self.trainmanager.aggrg,
            train_data=self.trainmanager.train_data,
            valid_data=self.trainmanager.valid_data,
            metrics=self.trainmanager.metrics,
            logger=self.trainmanager.logger,
            verbose=self.trainmanager.verbose,
        )
        self.trainmanager.make_private(message)

    async def _initialize_fairness(
        self,
    ) -> None:
        """Set up a fairness-enforcing algorithm as part of initialization.

        This method is optionally called in the context of `initialize`
        and should never be called in another context.
        """
        assert self.trainmanager is not None
        # Optionally setup SecAgg; await a FairnessSetupQuery.
        try:
            # When SecAgg is to be used, setup controllers first.
            if self.secagg is not None:
                received = await self.netwk.recv_message()
                await self.setup_secagg(received)
            # Await and deserialize a FairnessSetupQuery.
            received = await self.netwk.recv_message()
            query = await verify_server_message_validity(
                self.netwk, received, expected=messaging.FairnessSetupQuery
            )
        except Exception as exc:
            error = f"Fairness initialization failed: {repr(exc)}."
            self.logger.critical(error)
            raise RuntimeError(error) from exc
        self.logger.info("Received fairness setup instructions.")
        # Instantiate a FairnessControllerClient and run its setup routine.
        try:
            self.fairness = FairnessControllerClient.from_setup_query(
                query=query, manager=self.trainmanager
            )
            await self.fairness.setup_fairness(
                netwk=self.netwk, secagg=self._encrypter
            )
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
        message: messaging.TrainRequest,
    ) -> None:
        """Run a local training round.

        If an exception is raised during the local process, wrap
        it as an Error message and send it to the server instead
        of raising it.

        Parameters
        ----------
        message: TrainRequest
            Instructions from the server regarding the training round.
        """
        assert self.trainmanager is not None
        # When SecAgg is to be used, verify that it was set up.
        if self.secagg is not None and self._encrypter is None:
            error = (
                f"Refusing to participate in training round {message.round_i}"
                "as SecAgg is configured to be used but was not set up."
            )
            self.logger.error(error)
            await self.netwk.send_message(messaging.Error(error))
            return
        # Run the training round.
        reply: Message = self.trainmanager.training_round(message)
        # Collect and optionally record batch-wise training losses.
        # Note: collection enables purging them from memory.
        losses = self.trainmanager.model.collect_training_losses()
        if self.ckptr is not None:
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
                cleartext=reply, encrypter=self._encrypter
            )
        # Send training results (or error message) to the server.
        await self.netwk.send_message(reply)

    async def evaluation_round(
        self,
        message: messaging.EvaluationRequest,
    ) -> None:
        """Run a local evaluation round.

        If an exception is raised during the local process, wrap
        it as an Error message and send it to the server instead
        of raising it.

        If a checkpointer is set, record the local loss, and the
        model weights received from the server.

        Parameters
        ----------
        message: EvaluationRequest
            Instructions from the server regarding the evaluation round.
        """
        assert self.trainmanager is not None
        # When SecAgg is to be used, verify that it was set up.
        if self.secagg is not None and self._encrypter is None:
            error = (
                "Refusing to participate in evaluation round "
                f"{message.round_i} as SecAgg is configured to be used "
                "but was not set up."
            )
            self.logger.error(error)
            await self.netwk.send_message(messaging.Error(error))
            return
        # Run the evaluation round.
        reply: Message = self.trainmanager.evaluation_round(message)
        # Post-process the results.
        if isinstance(reply, messaging.EvaluationReply):  # not an Error
            # Optionnally checkpoint the model, optimizer and local loss.
            if self.ckptr:
                self.ckptr.checkpoint(
                    model=self.trainmanager.model,
                    optimizer=self.trainmanager.optim,
                    metrics=self.trainmanager.metrics.get_result(),
                )
            # Optionally prevent sharing metrics (save for the loss).
            if not self.share_metrics:
                reply.metrics.clear()
            # Optionally SecAgg-encrypt results.
            if self._encrypter is not None:
                msg_cls = secagg_messaging.SecaggEvaluationReply
                reply = msg_cls.from_cleartext_message(
                    cleartext=reply, encrypter=self._encrypter
                )
        # Send evaluation results (or error message) to the server.
        await self.netwk.send_message(reply)

    async def fairness_round(
        self,
        query: messaging.FairnessQuery,
    ) -> None:
        """Handle a server request to run a fairness-related round.

        The nature of the round depends on the fairness-aware learning
        algorithm that was optionally set up during the initialization
        phase. In case no such algorithm was set up, this method will
        raise a process-crashing exception.

        Parameters
        ----------
        query:
            `FairnessQuery` message from the server.

        Raises
        ------
        RuntimeError
            If no fairness controller was set up for this instance.
        """
        assert self.trainmanager is not None
        # If no fairness controller was set up, raise a RuntimeError.
        if self.fairness is None:
            error = (
                "Received a query to participate in a fairness round, "
                "but no fairness controller was set up."
            )
            self.logger.critical(error)
            await self.netwk.send_message(messaging.Error(error))
            raise RuntimeError(error)
        # When SecAgg is to be used, verify that it was set up.
        if self.secagg is not None and self._encrypter is None:
            error = (
                "Refusing to participate in fairness-related round "
                f"{query.round_i} as SecAgg is configured to be used "
                "but was not set up."
            )
            self.logger.error(error)
            await self.netwk.send_message(messaging.Error(error))
            return
        # Otherwise, run the controller's routine.
        metrics = await self.fairness.run_fairness_round(
            netwk=self.netwk, query=query, secagg=self._encrypter
        )
        # Optionally save computed fairness metrics.
        # similar to server code; pylint: disable=duplicate-code
        if self.ckptr is not None:
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
        self.logger.info(
            "Training is now over, after %s rounds. Global loss: %s",
            message.rounds,
            message.loss,
        )
        if self.ckptr:
            path = os.path.join(self.ckptr.folder, "model_state_best.json")
            self.logger.info("Checkpointing final weights under %s.", path)
            assert self.trainmanager is not None  # for mypy
            self.trainmanager.model.set_weights(message.weights)
            self.ckptr.save_model(self.trainmanager.model, timestamp="best")

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
