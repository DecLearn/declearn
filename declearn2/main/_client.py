# coding: utf-8

"""Client-side main Federated Learning orchestrating class."""

import asyncio
import dataclasses
import json
import os
import time
from typing import Any, Dict, Optional, Tuple


from declearn2.communication import messaging
from declearn2.communication.api import Client
from declearn2.dataset import Dataset
from declearn2.model.api import Model
from declearn2.optimizer import Optimizer
from declearn2.utils import get_logger, json_pack, serialize_object


class FederatedClient:
    """Client-side Federated Learning orchestrating class."""

    logger = get_logger("federated-client")

    def __init__(
            self,
            client: Client,  # revise: from_config
            dataset: Dataset,  # revise: from_json
            folder: str,
        ) -> None:
        """Docstring."""
        self.netwk = client
        self.dataset = dataset
        self.folder = folder

    def run(
            self,
        ) -> None:
        """Docstring."""
        asyncio.run(self.training())

    async def training(
            self,
        ) -> None:
        """Participate in the federated training process."""
        async with self.netwk:
            # Register for training, then collect initialization information.
            await self.register()
            model, optim = await self.initialize()
            # Process server instructions as they come.
            while True:
                message = await self.netwk.check_message()
                if isinstance(message, messaging.TrainRequest):
                    await self.training_round(model, optim, message)
                elif isinstance(message, messaging.EvaluationRequest):
                    await self.evaluation_round(model, message)
                elif isinstance(message, messaging.StopTraining):
                    await self.stop_training(model, message)
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
        data_info = dataclasses.asdict(self.dataset.get_data_specs())
        for i in range(10):  # max_attempts (10)
            self.logger.info("Attempting to join training (trial n°%s)", i)
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
            start_weights = model.get_weights()
            # Train under instructed effort constraints.
            self.logger.info(
                "Training local model for %s epochs | %s steps | %s seconds.",
                message.n_epoch, message.n_steps, message.timeout
            )
            effort = self._train_for(
                model, optim, message.n_epoch, message.n_steps,
                message.timeout, batch_size=message.batch_s,
                # revise: enable passing other dataset arguments?
            )
            # Compute model updates and collect auxiliary variables.
            self.logger.info("Sending local updates to the server.")
            reply = messaging.TrainReply(
                updates=start_weights - model.get_weights(),
                aux_var=optim.collect_aux_var(),
                **effort
            )  # type: messaging.Message
        # In case of failure, ensure it is reported to the server.
        except Exception as exception:  # pylint: disable=broad-except
            reply = messaging.Error(repr(exception))
        # Send training results (or error message) to the server.
        await self.netwk.send_message(reply)

    def _train_for(
            self,
            model: Model,
            optim: Optimizer,
            epochs: Optional[int] = None,
            steps: Optional[int] = None,
            timeout: Optional[int] = None,
            **kwargs: Any,
        )  -> Dict[str, int]:
        """Backend code to run local SGD steps under effort constraints.

        Parameters
        ----------
        model:
            Model that is to be trained locally.
        optim:
            Optimizer to be used when computing local SGD steps.
        epochs: int or None, default=None
            Maximum number of local training epochs to perform.
            May be overridden by `steps` or `timeout`.
        steps: int or None, default=None
            Maximum number of local training steps to perform.
            May be overridden by `epochs` or `timeout`.
        timeout: int or None, default=None
            Time (in seconds) beyond which to interrupt training,
            regardless of the actual number of steps taken (> 0).
        **kwargs:
            Keyword arguments to `self.dataset.generate_batches`
            may also be passed to this function, e.g. to specify
            the `batch_size` of local SGD steps.

        Returns
        -------
        effort: dict[str, int]
            Dictionary storing information on the computational
            effort effectively performed:
            * n_epoch: int
                Number of training epochs completed.
            * n_steps: int
                Number of training steps completed.
            * t_spent: int
                Time spent running training steps (in seconds).
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        if all(arg is None for arg in (epochs, steps, timeout)):
            raise ValueError("At least one control argument must be set.")
        if epochs is not None:
            epochs = max(epochs, 1)
        if steps is None:
            steps = float("inf")  # type: ignore
        if timeout is None:
            timeout = float("inf")  # type: ignore
        # Set up variables to keep track of constaints and efforts.
        t_start = time.time()
        t_spent = 0.
        n_epoch = 0
        n_steps = 0
        # Run batch train steps for as long as constraints set it.
        while (epochs is None) or (n_epoch < epochs):
            for batch in self.dataset.generate_batches(**kwargs):
                optim.run_train_step(model, batch)
                n_steps += 1
                t_spent = time.time() - t_start
                if (n_steps >= steps) or (t_spent >= timeout):  # type: ignore
                    break
            n_epoch += 1
        # Return a dict storing information on the training effort.
        return {
            "n_epoch": n_epoch, "n_steps": n_steps, "t_spent": round(t_spent)
        }

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
            data = self.dataset.generate_batches(batch_size=message.batch_s)
            loss = model.compute_loss(data)
            nstp = self.dataset.get_data_specs().n_samples // message.batch_s
            # future: implement the former more elegantly
            reply = messaging.EvaluationReply(
                loss=loss, n_steps=nstp
            )  # type: messaging.Message
        # In case of failure, ensure it is reported to the server.
        except Exception as exception:  # pylint: disable=broad-except
            self.logger.error(
                "Error encountered during evaluation: %s.", exception
            )
            reply = messaging.Error(repr(exception))
        # Send training results (or error message) to the server.
        await self.netwk.send_message(reply)

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
        model.set_weights(message.weights)
        dump = {
            "model": serialize_object(model),
            "weights": model.get_weights(),
        }
        path = os.path.join(self.folder, "final_model.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(dump, file, default=json_pack)

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
