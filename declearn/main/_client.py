# coding: utf-8

"""Client-side main Federated Learning orchestrating class."""

import asyncio
import dataclasses
import json
import os
from typing import Any, Dict, Optional, Tuple, Union


from declearn.communication import NetworkClientConfig, messaging
from declearn.communication.api import Client
from declearn.dataset import Dataset, load_dataset_from_json
from declearn.main.utils import (
    Checkpointer, Constraint, ConstraintSet, TimeoutConstraint
)
from declearn.model.api import Model
from declearn.optimizer import Optimizer
from declearn.utils import get_logger, json_pack


__all__ = [
    'FederatedClient',
]


class FederatedClient:
    """Client-side Federated Learning orchestrating class."""

    logger = get_logger("federated-client")

    def __init__(
            self,
            netwk: Union[Client, NetworkClientConfig, Dict[str, Any]],
            train_data: Union[Dataset, str],
            valid_data: Optional[Union[Dataset, str]] = None,
            folder: Optional[str] = None,
        ) -> None:
        """Instantiate a client to participate in a federated learning task.

        Parameters
        ----------
        netwk: Client or NetworkClientConfig or dict
            Client communication endpoint instance, or configuration
            dict or dataclass enabling its instantiation.
        train_data: Dataset or str
            Dataset instance wrapping the training data, or path to
            a JSON file from which it can be instantiated.
        valid_data: Dataset or str or None
            Optional Dataset instance wrapping validation data, or
            path to a JSON file from which it can be instantiated.
            If None, run evaluation rounds over `train_data`.
        folder: str or None, default=None
            Optional folder where to write out a model dump, round-
            wise weights checkpoints and local validation losses.
            If None, only record the loss metric and lowest-loss-
            yielding weights in memory (under `self.checkpoint`).
        """
        # Assign the wrapped communication Client.
        if isinstance(netwk, dict):
            netwk = NetworkClientConfig(**netwk).build_client()
        elif isinstance(netwk, NetworkClientConfig):
            netwk = netwk.build_client()
        elif not isinstance(netwk, Client):
            raise TypeError(
                "'netwk' should be a declearn.communication.Client, "
                "or the valid configuration of one."
            )
        self.netwk = netwk
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
        # Record the checkpointing folder and create a Checkpointer slot.
        self.folder = folder
        self.checkpointer = None  # type: Optional[Checkpointer]

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
            model, optim = await self.initialize()
            # Instantiate a checkpointer and save the initial model.
            self.checkpointer = Checkpointer(model, self.folder)
            self.checkpointer.save_model()
            self.checkpointer.checkpoint(float("inf"))  # initial weights
            # Process server instructions as they come.
            while True:
                message = await self.netwk.check_message()
                if isinstance(message, messaging.TrainRequest):
                    await self.training_round(model, optim, message)
                elif isinstance(message, messaging.EvaluationRequest):
                    await self.evaluation_round(model, message)
                elif isinstance(message, messaging.StopTraining):
                    await self.stop_training(model, message)
                    break
                elif isinstance(message, messaging.CancelTraining):
                    await self.cancel_training(message)
                else:
                    error = "Unexpected instruction received from server:"
                    error += repr(message)
                    self.logger.error(error)
                    raise ValueError(error)

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
            self.logger.info("Attempting to join training (attempt n°%s)", i+1)
            registered = await self.netwk.register(data_info)
            if registered:
                break
            await asyncio.sleep(60)  # delay_retries (1 minute)
        else:
            raise RuntimeError("Failed to register for training.")

    async def initialize(
            self,
        ) -> Tuple[Model, Optimizer]:
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
        # Return the model and optimizer received from the server.
        return message.model, message.optim

    async def training_round(
            self,
            model: Model,
            optim: Optimizer,
            message: messaging.TrainRequest,
        ) -> None:
        """Run a local training round.

        If an exception is raised during the local process, wrap
        it as an Error message and send it to the server instead
        of raising it.

        Parameters
        ----------
        model:
            Model that is to be trained locally.
        optim:
            Optimizer to be used when computing local SGD steps.
        message: TrainRequest
            Instructions from the server regarding the training round.
        """
        self.logger.info("Participating in training round %s", message.round_i)
        # Try running the training round.
        try:
            # Unpack and apply model weights and optimizer auxiliary variables.
            self.logger.info("Applying server updates to local objects.")
            model.set_weights(message.weights)
            optim.process_aux_var(message.aux_var)
            # Train under instructed effort constraints.
            self.logger.info(
                "Training local model for %s epochs | %s steps | %s seconds.",
                message.n_epoch, message.n_steps, message.timeout
            )
            effort = self._train_under_constraints(
                model, optim, message.batches,
                message.n_epoch, message.n_steps, message.timeout,
            )
            # Compute model updates and collect auxiliary variables.
            self.logger.info("Sending local updates to the server.")
            reply = messaging.TrainReply(
                updates=message.weights - model.get_weights(),
                aux_var=optim.collect_aux_var(),
                n_epoch=int(effort["n_epoch"]),
                n_steps=int(effort["n_steps"]),
                t_spent=round(effort["t_spent"], 3),
            )  # type: messaging.Message
        # In case of failure, ensure it is reported to the server.
        except Exception as exception:  # pylint: disable=broad-except
            reply = messaging.Error(repr(exception))
        # Send training results (or error message) to the server.
        await self.netwk.send_message(reply)

    def _train_under_constraints(
            self,
            model: Model,
            optim: Optimizer,
            batch_cfg: Dict[str, Any],
            n_epoch: Optional[int],
            n_steps: Optional[int],
            timeout: Optional[int],
        ) -> Dict[str, float]:
        """Backend code to run local SGD steps under effort constraints.

        Parameters
        ----------
        model:
            Model that is to be trained locally.
        optim:
            Optimizer to be used when computing local SGD steps.
        batch_cfg: Dict[str, Any]
            Keyword arguments for `self.train_data.generate_batches`
            i.e. specifications of batches used in local SGD steps.
        n_epoch: int or None, default=None
            Maximum number of local training epochs to perform.
            May be overridden by `n_steps` or `timeout`.
        n_steps: int or None, default=None
            Maximum number of local training steps to perform.
            May be overridden by `n_epoch` or `timeout`.
        timeout: int or None, default=None
            Time (in seconds) beyond which to interrupt training,
            regardless of the actual number of steps taken (> 0).

        Returns
        -------
        effort: dict[str, float]
            Dictionary storing information on the computational
            effort effectively performed:
            * n_epoch: int
                Number of training epochs completed.
            * n_steps: int
                Number of training steps completed.
            * t_spent: float
                Time spent running training steps (in seconds).
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        # Set up effort constraints under which to operate.
        epochs = Constraint(limit=n_epoch, name="n_epoch")
        constraints = ConstraintSet([
            Constraint(limit=n_steps, name="n_steps"),
            TimeoutConstraint(limit=timeout, name="t_spent"),
        ])
        # Run batch train steps for as long as constraints allow it.
        while not epochs.saturated:
            for batch in self.train_data.generate_batches(**batch_cfg):
                optim.run_train_step(model, batch)
                constraints.increment()
                if constraints.saturated:
                    break
            epochs.increment()
        # Return a dict storing information on the training effort.
        effort = {"n_epoch": epochs.value}
        effort.update(constraints.get_values())
        return effort

    async def evaluation_round(
            self,
            model: Model,
            message: messaging.EvaluationRequest,
        ) -> None:
        """Run a local evaluation round.

        If an exception is raised during the local process, wrap
        it as an Error message and send it to the server instead
        of raising it.

        Parameters
        ----------
        model:
            Model that is to be evaluated locally.
        message: EvaluationRequest
            Instructions from the server regarding the evaluation round.
        """
        self.logger.info(
            "Participating in evaluation round %s", message.round_i
        )
        # Try running the evaluation round.
        try:
            # Update the model's weights and evaluate on the local dataset.
            model.set_weights(message.weights)
            reply = self._evaluate_under_constraints(
                model, message.batches, message.n_steps, message.timeout
            )
            # If possible, checkpoint the model and record the local loss.
            if self.checkpointer is not None:  # True in `run` context
                self.checkpointer.checkpoint(reply.loss)
        # In case of failure, ensure it is reported to the server.
        except Exception as exception:  # pylint: disable=broad-except
            self.logger.error(
                "Error encountered during evaluation: %s.", exception
            )
            reply = messaging.Error(repr(exception))  # type: ignore
        # Send training results (or error message) to the server.
        await self.netwk.send_message(reply)

    def _evaluate_under_constraints(
            self,
            model: Model,
            batch_cfg: Dict[str, Any],
            n_steps: Optional[int] = None,
            timeout: Optional[int] = None,
        )  -> messaging.EvaluationReply:
        """Backend code to run local loss computation under effort constraints.

        Parameters
        ----------
        model:
            Model that is to be evaluated locally.
        batch_cfg: Dict[str, Any]
            Keyword arguments to `self.valid_data.generate_batches`.
        n_steps: int or None, default=None
            Maximum number of local evaluation steps to perform.
            May be overridden by `timeout` or dataset size.
        timeout: int or None, default=None
            Time (in seconds) beyond which to interrupt evaluation,
            regardless of the actual number of steps taken (> 0).

        Returns
        -------
        eval_reply: messaging.EvaluationReply
            EvaluationReply message wrapping the computed loss on the
            local validation (or, if absent, training) dataset as well
            as the number of steps and the time taken to obtain it.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        # Set up effort constraints under which to operate.
        constraints = ConstraintSet([
            Constraint(limit=n_steps, name="n_steps"),
            TimeoutConstraint(limit=timeout, name="t_spent"),
        ])
        # Run batch evaluation steps for as long as constraints allow it.
        loss = 0.
        dataset = self.valid_data or self.train_data
        for batch in dataset.generate_batches(**batch_cfg):
            loss += model.compute_loss([batch])
            constraints.increment()
            if constraints.saturated:
                break
        # Pack the result and computational effort information into a message.
        effort = constraints.get_values()
        return messaging.EvaluationReply(
            loss=loss / effort["n_steps"],
            n_steps=int(effort["n_steps"]),
            t_spent=round(effort["t_spent"], 3),
        )

    async def stop_training(
            self,
            model: Model,
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
            message.rounds, message.loss
        )
        if self.folder is not None:
            # Save the locally-best-performing model weights.
            if self.checkpointer is not None:  # True in `run` context
                path = os.path.join(self.folder, "best_local_weights.json")
                self.logger.info("Saving best local weights in '%s'.", path)
                self.checkpointer.reset_best_weights()
                with open(path, "w", encoding="utf-8") as file:
                    json.dump(model.get_weights(), file, default=json_pack)
            # Save the globally-best-performing model weights.
            path = os.path.join(self.folder, "final_weights.json")
            self.logger.info("Saving final weights in '%s'.", path)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(message.weights, file, default=json_pack)

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
        #self.logger.info("Saving the current model and optimizer.")
        raise RuntimeError(error)
