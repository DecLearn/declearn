# coding: utf-8

"""Client-side main Federated Learning orchestrating class."""

import asyncio
import dataclasses
from typing import Any, Dict, Tuple


from declearn2.communication.api import Client
from declearn2.communication.api.flags import FLAG_WELCOME
from declearn2.dataset import Dataset
from declearn2.model.api import Model
from declearn2.optimizer import Optimizer
from declearn2.utils import deserialize_object, get_logger


# revise: move this to shared flags
INITIALIZE = "initialize from sent params"
CANCEL_TRAINING = "training is cancelled"
TRAIN_ONE_ROUND = "train for one round"


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
            action, params = await self.netwk.check_message()
            if action == TRAIN_ONE_ROUND:
                await self._train_one_round(model, optim, params)
            elif action == CANCEL_TRAINING:
                await self._cancel_training(params)
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
            reply = await self.netwk.register(data_info)
            if reply == FLAG_WELCOME:
                break
            await asyncio.sleep(60)  # delay_retries (1 minute)
        else:
            raise RuntimeError("Failed to register for training.")

    async def initialize(
            self,
        ) -> Tuple[Model, Optimizer]:
        """Docstring."""
        # Await initialization instructions.
        action, params = await self.netwk.check_message()
        if action != INITIALIZE:
            error = f"Awaited INITIALIZE instructions, got: '{params}'"
            self.logger.error(error)
            raise RuntimeError(error)
        for key in ("model", "optim"):
            if key not in params:
                error = f"Missing required initialization key: '{key}'."
                self.logger.error(error)
                raise KeyError(error)
        # Instantiate the model and optimizer, and return them.
        model = deserialize_object(params["model"])
        if not isinstance(model, Model):
            error = "Received model was not deserialized as a Model."
            self.logger.error("Fatal Initialization Error: %s", error)
            await self.netwk.send_message({"error": error})
            raise RuntimeError(error)
        optim = deserialize_object(params["optim"])
        if not isinstance(optim, Optimizer):
            error = "Received optim was not deserialized as an Optimizer."
            self.logger.error("Fatal Initialization Error: %s", error)
            await self.netwk.send_message({"error": error})
            raise RuntimeError(error)
        return model, optim

    async def _train_one_round(
            self,
            model: Model,
            optim: Optimizer,
            params: Dict[str, Any],
        ) -> None:
        """Docstring."""
        message = {
            "updates": None,
            "aux_var": None,
            "n_steps": 0,
            "error": None,
        }  # type: Dict[str, Any]
        # Try running the training round.
        try:  # revise: use a dataclass to unpack params
            self.logger.info(
                "Participating in training round %s", params["round_i"]
            )
            # Unpack and apply model weights and optimizer auxiliary variables.
            self.logger.info("Applying server updates to local objects.")
            model.set_weights(params["weights"])
            optim.process_aux_var(params["aux_var"])
            start_weights = model.get_weights()
            # Train for a full epoch.
            self.logger.info("Training local model for 1 epoch.")
            batches = self.dataset.generate_batches(
                batch_size=params["batch_size"],
                # revise: improve kwargs handling
            )
            for batch in batches:  # revise: add more modularity to training
                optim.run_train_step(model, batch)
                message["n_steps"] += 1
            # Compute model updates and collect auxiliary variables.
            message["updates"] = start_weights - model.get_weights()
            message["aux_var"] = optim.collect_aux_var()
        # In case of failure, ensure it is reported to the server.
        except Exception as exception:  # pylint: disable=broad-except
            message["error"] = repr(exception)
        # Send training results (or error message) to the server.
        await self.netwk.send_message(message)

    async def _cancel_training(
            self,
            params: Dict[str, Any],
        ) -> None:
        """Docstring."""
        error = "Training was cancelled by the server, with reason:\n"
        error += params.get("reason", "(error: unspecified reason)")
        self.logger.warning(error)
        #self.logger.info("Saving the current model and optimizer.")
        raise RuntimeError(error)
