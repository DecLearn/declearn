# coding: utf-8

"""Client-side main Federated Learning orchestrating class."""

import asyncio
import dataclasses
import time
from typing import Any, Dict, Optional, Tuple


from declearn2.communication import messaging
from declearn2.communication.api import Client
from declearn2.dataset import Dataset
from declearn2.model.api import Model
from declearn2.optimizer import Optimizer
from declearn2.utils import get_logger


class FederatedClient:
    """Client-side Federated Learning orchestrating class."""

    logger = get_logger("federated-client")

    def __init__(
            self,
            client: Client,  # revise: from_config
            dataset: Dataset,  # revise: from_json
        ) -> None:
        """Docstring."""
        self.netwk = client
        self.dataset = dataset

    def run(
            self,
        ) -> None:
        """Docstring."""
        self.netwk.run_until_complete(self.training)

    async def training(
            self,
        ) -> None:
        """Participate in the federated training process."""
        # Register for training, then collect initialization information.
        await self.register()
        model, optim = await self.initialize()
        # Process server instructions as they come.
        while True:
            message = await self.netwk.check_message()
            if isinstance(message, messaging.TrainRequest):
                await self._train_one_round(model, optim, message)
            elif isinstance(message, messaging.CancelTraining):
                await self._cancel_training(message)
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
        await self.netwk.send_message(messaging.Empty())
        # Return the model and optimizer received from the server.
        return message.model, message.optim

    async def _train_one_round(
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
        # Try running the training round.
        try:  # revise: use a dataclass to unpack params
            self.logger.info(
                "Participating in training round %s", message.round_i
            )
            # Unpack and apply model weights and optimizer auxiliary variables.
            self.logger.info("Applying server updates to local objects.")
            model.set_weights(message.weights)  # type: ignore
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

    async def _cancel_training(
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
