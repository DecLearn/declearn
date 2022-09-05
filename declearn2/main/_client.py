# coding: utf-8

"""Client-side main Federated Learning orchestrating class."""

import asyncio
import dataclasses
from typing import Tuple


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
        """Docstring."""
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
                error = "Unexpected instruction received from server."
                self.logger.error(error)
                raise ValueError(error)

    async def register(
            self,
        ) -> None:
        """Docstring."""
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
        """Docstring."""
        # Await initialization instructions.
        message = await self.netwk.check_message()
        if not isinstance(message, messaging.InitRequest):
            error = f"Awaited InitRequest message, got: '{message}'"
            self.logger.error(error)
            raise RuntimeError(error)
        # Return the model and optimizer received from the server.
        return message.model, message.optim

    async def _train_one_round(
            self,
            model: Model,
            optim: Optimizer,
            message: messaging.TrainRequest,
        ) -> None:
        """Docstring."""
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
            # Train for a full epoch.  # TODO: modularize using message fields
            self.logger.info("Training local model for 1 epoch.")
            batches = self.dataset.generate_batches(
                batch_size=message.batch_s,
                # revise: improve kwargs handling
            )
            n_steps = 0
            for batch in batches:
                optim.run_train_step(model, batch)
                n_steps += 1
            # Compute model updates and collect auxiliary variables.
            reply = messaging.TrainReply(
                n_epoch=1,
                n_steps=n_steps,
                t_spent=0,  # fixme: implement
                updates=start_weights - model.get_weights(),
                aux_var=optim.collect_aux_var(),
            )  # type: messaging.Message
        # In case of failure, ensure it is reported to the server.
        except Exception as exception:  # pylint: disable=broad-except
            reply = messaging.Error(repr(exception))
        # Send training results (or error message) to the server.
        await self.netwk.send_message(reply)

    async def _cancel_training(
            self,
            message: messaging.CancelTraining,
        ) -> None:
        """Docstring."""
        error = "Training was cancelled by the server, with reason:\n"
        error += message.reason
        self.logger.warning(error)
        #self.logger.info("Saving the current model and optimizer.")
        raise RuntimeError(error)
