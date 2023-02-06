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

"""Client-side main Federated Learning orchestrating class."""

import asyncio
import dataclasses
import logging
from typing import Any, Dict, Optional, Union

from declearn.communication import NetworkClientConfig, messaging
from declearn.communication.api import NetworkClient
from declearn.dataset import Dataset, load_dataset_from_json
from declearn.main.utils import Checkpointer, TrainingManager
from declearn.utils import get_logger


__all__ = [
    "FederatedClient",
]


class FederatedClient:
    """Client-side Federated Learning orchestrating class."""

    # one-too-many attribute; pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        netwk: Union[NetworkClient, NetworkClientConfig, Dict[str, Any], str],
        train_data: Union[Dataset, str],
        valid_data: Optional[Union[Dataset, str]] = None,
        checkpoint: Union[Checkpointer, Dict[str, Any], str, None] = None,
        share_metrics: bool = True,
        logger: Union[logging.Logger, str, None] = None,
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
            Dataset instance wrapping the training data, or path to
            a JSON file from which it can be instantiated.
        valid_data: Dataset or str or None
            Optional Dataset instance wrapping validation data, or
            path to a JSON file from which it can be instantiated.
            If None, run evaluation rounds over `train_data`.
        checkpoint: Checkpointer or dict or str or None, default=None
            Optional Checkpointer instance or instantiation dict to be
            used so as to save round-wise model, optimizer and metrics.
            If a single string is provided, treat it as the checkpoint
            folder path and use default values for other parameters.
        share_metrics: bool, default=True
            Whether to share evaluation metrics with the server,
            or save them locally and only send the model's loss.
            This may prevent information leakage, e.g. as to the
            local distribution of target labels or values.
        logger: logging.Logger or str or None, default=None,
            Logger to use, or name of a logger to set up with
            `declearn.utils.get_logger`.
            If None, use `type(self):netwk.name`.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        # Assign the wrapped NetworkClient.
        replace_netwk_logger = False
        if isinstance(netwk, str):
            netwk = NetworkClientConfig.from_toml(netwk)
        elif isinstance(netwk, dict):
            netwk = NetworkClientConfig.from_params(**netwk)
        if isinstance(netwk, NetworkClientConfig):
            replace_netwk_logger = netwk.logger is None
            netwk = netwk.build_client()
        if not isinstance(netwk, NetworkClient):
            raise TypeError(
                "'netwk' should be a declearn.communication.api.NetworkClient,"
                " or the valid configuration of one."
            )
        self.netwk = netwk
        # Assign the logger and optionally replace that of the network client.
        if not isinstance(logger, logging.Logger):
            logger = get_logger(
                logger or f"{type(self).__name__}-{netwk.name}"
            )
        self.logger = logger
        if replace_netwk_logger:
            self.netwk.logger = self.logger
        # Assign the wrapped training dataset.
        if isinstance(train_data, str):
            train_data = load_dataset_from_json(train_data)
        if not isinstance(train_data, Dataset):
            raise TypeError("'train_data' should be a Dataset or path to one.")
        self.train_data = train_data
        # Assign the wrapped validation dataset (if any).
        if isinstance(valid_data, str):
            valid_data = load_dataset_from_json(valid_data)
        if not (valid_data is None or isinstance(valid_data, Dataset)):
            raise TypeError("'valid_data' should be a Dataset or path to one.")
        self.valid_data = valid_data
        # Assign an optional checkpointer.
        if checkpoint is not None:
            checkpoint = Checkpointer.from_specs(checkpoint)
        self.ckptr = checkpoint
        # Record the metric-sharing boolean switch.
        self.share_metrics = bool(share_metrics)
        # Create a TrainingManager slot, populated at initialization phase.
        self.trainmanager = None  # type: Optional[TrainingManager]

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
                message = await self.netwk.check_message()
                stoprun = await self.handle_message(message)
                if stoprun:
                    break

    async def handle_message(
        self,
        message: messaging.Message,
    ) -> bool:
        """Handle an incoming message from the server.

        Parameters
        ----------
        message: messaging.Message
            Message instance that needs triage and processing.

        Returns
        -------
        exit_loop: bool
            Whether to interrupt the client's message-receiving loop.
        """
        exit_loop = False
        if isinstance(message, messaging.TrainRequest):
            await self.training_round(message)
        elif isinstance(message, messaging.EvaluationRequest):
            await self.evaluation_round(message)
        elif isinstance(message, messaging.StopTraining):
            await self.stop_training(message)
            exit_loop = True
        elif isinstance(message, messaging.CancelTraining):
            await self.cancel_training(message)
        else:
            error = "Unexpected instruction received from server:"
            error += repr(message)
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
        # revise: add validation dataset specs
        data_info = dataclasses.asdict(self.train_data.get_data_specs())
        for i in range(10):  # max_attempts (10)
            self.logger.info(
                "Attempting to join training (attempt n°%s)", i + 1
            )
            registered = await self.netwk.register(data_info)
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
        RuntimeError:
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
        # Await initialization instructions. Report messages-unpacking errors.
        self.logger.info("Awaiting initialization instructions from server.")
        try:
            message = await self.netwk.check_message()
        except Exception as exc:
            await self.netwk.send_message(messaging.Error(repr(exc)))
            raise RuntimeError("Initialization failed.") from exc
        # Otherwise, check that the request is of valid type.
        if not isinstance(message, messaging.InitRequest):
            error = f"Awaited InitRequest message, got: '{message}'"
            self.logger.error(error)
            raise RuntimeError(error)
        # Send back an empty message to indicate that all went fine.
        self.logger.info("Notifying the server that initialization went fine.")
        await self.netwk.send_message(
            messaging.GenericMessage(action="InitializationOK", params={})
        )
        # Wrap up the model and optimizer received from the server.
        self.trainmanager = TrainingManager(
            model=message.model,
            optim=message.optim,
            train_data=self.train_data,
            valid_data=self.valid_data,
            metrics=message.metrics,
            logger=self.logger,
        )
        # If instructed to do so, await a PrivacyRequest to set up DP-SGD.
        if message.dpsgd:
            await self._initialize_dpsgd()
        # Optionally checkpoint the received model and optimizer.
        if self.ckptr:
            self.ckptr.checkpoint(
                model=self.trainmanager.model,
                optimizer=self.trainmanager.optim,
                first_call=True,
            )

    async def _initialize_dpsgd(
        self,
    ) -> None:
        """Set up differentially-private training as part of initialization.

        This method wraps the `make_private` one in the context of
        `initialize` and should never be called in another context.
        """
        message = await self.netwk.check_message()
        if not isinstance(message, messaging.PrivacyRequest):
            msg = f"Expected a PrivacyRequest but received a '{type(message)}'"
            self.logger.error(msg)
            await self.netwk.send_message(messaging.Error(msg))
            raise RuntimeError(f"DP-SGD initialization failed: {msg}.")
        self.logger.info("Received a request to set up DP-SGD.")
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
        await self.netwk.send_message(
            messaging.GenericMessage(action="privacy-ok", params={})
        )

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
        from declearn.main.privacy import DPTrainingManager

        # pylint: enable=import-outside-toplevel
        self.trainmanager = DPTrainingManager(
            model=self.trainmanager.model,
            optim=self.trainmanager.optim,
            train_data=self.trainmanager.train_data,
            valid_data=self.trainmanager.valid_data,
            metrics=self.trainmanager.metrics,
            logger=self.trainmanager.logger,
        )
        self.trainmanager.make_private(message)

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
        manager: TrainingManager
            Instance wrapping the model, optimizer and data to use.
        message: TrainRequest
            Instructions from the server regarding the training round.
        """
        assert self.trainmanager is not None
        # Run the training round.
        reply = self.trainmanager.training_round(message)
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
        manager: TrainingManager
            Instance wrapping the model and data to use.
        message: EvaluationRequest
            Instructions from the server regarding the evaluation round.
        """
        assert self.trainmanager is not None
        # Run the evaluation round.
        reply = self.trainmanager.evaluation_round(message)
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
        # Send evaluation results (or error message) to the server.
        await self.netwk.send_message(reply)

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
            path = f"{self.ckptr.folder}/model_state_best.json"
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
