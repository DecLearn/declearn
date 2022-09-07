# coding: utf-8

"""Server-side main Federated Learning orchestrating class."""

import asyncio
from typing import Any, Dict, Set


from declearn2.communication import messaging
from declearn2.communication.api import Server
from declearn2.main._data_info import (
    AggregationError, aggregate_clients_data_info
)
from declearn2.model.api import Model
from declearn2.strategy import Strategy
from declearn2.utils import get_logger


class FederatedServer:
    """Server-side Federated Learning orchestrating class."""

    logger = get_logger("federated-server")

    def __init__(
            self,
            model: Model,    # revise: from_config
            server: Server,  # revise: from_config
            strategy: Strategy,  # revise: from_config
            batch_size: int,
        ) -> None:
        """Docstring."""
        self.model = model
        self.netwk = server
        self.strat = strategy
        self.aggrg = self.strat.build_server_aggregator()
        self.optim = self.strat.build_server_optimizer()
        self.batch_size = batch_size

    def run(
            self,
            rounds: int,
        ) -> None:
        """Docstring."""
        asyncio.run(self.training(rounds))

    async def training(
            self,
            rounds: int,
        ) -> None:
        """Orchestrate the federated training routine."""
        async with self.netwk:
            await self.initialization()
            round_i = 0
            while True:
                await self.training_round(round_i)
                round_i += 1
                if round_i >= rounds:
                    break

    async def initialization(
            self,
        ) -> None:
        """Orchestrate the initialization steps to set up training.

        Wait for clients to register and process their data information.
        Send instructions to clients to set up their model and optimizer.
        Await clients to have finalized their initialization step; raise
        and cancel training if issues are reported back.

        Raises
        ------
        RuntimeError:
            In case any of the clients returned an Error message rather
            than an Empty ping-back message. Send CancelTraining to all
            clients before raising.
        """
        # Wait for clients to register and process their data information.
        data_info = await self.netwk.wait_for_clients()  # revise: nb_clients
        await self._process_data_info(data_info)
        # Serialize intialization information and send it to clients.
        message = messaging.InitRequest(
            model=self.model,
            optim=self.strat.build_client_optimizer(),
        )  # revise: strategy rather than optimizer?
        await self.netwk.broadcast_message(message)
        # Await a confirmation from clients that initialization went well.
        replies = await self.netwk.wait_for_messages()
        errors = {
            client: msg.message
            for client, msg in replies.items()
            if isinstance(msg, messaging.Error)
        }
        # If any client has failed to initialize, raise.
        if errors:
            err_msg = "Initialization failed for another client."
            await self.netwk.send_messages({
                client: messaging.CancelTraining(errors.get(client, err_msg))
                for client in self.netwk.client_names
            })
            err_msg = f"Initialization failed for {len(errors)} clients:"
            err_msg += "".join(
                f"\n    {client}: {error}" for client, error in errors.items()
            )
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)

    async def _process_data_info(  # revise: drop async
            self,
            clients_data_info: Dict[str, Dict[str, Any]],
        ) -> None:
        """Validate, aggregate and make use of clients' data-info.

        Parameters
        ----------
        clients_data_info: dict[str, dict[str, any]]
            Client-wise data-info dict, that are to be aggregated
            and passed to the global model for initialization.

        Raises
        ------
        AggregationError:
            In case (some of) the clients' data info is invalid, or
            incompatible. Send CancelTraining to all clients before
            raising.
        """
        fields = self.model.required_data_info  # revise: add optimizer, etc.
        # Try aggregating the input data_info.
        try:
            info = aggregate_clients_data_info(clients_data_info, fields)
        # In case of failure, cancel training, notify clients, log and raise.
        except AggregationError as exc:
            await self.netwk.send_messages({
                client: messaging.CancelTraining(reason)
                for client, reason in exc.messages.items()
            })
            self.logger.error(exc.error)
            raise exc
        # Otherwise, initialize the model based on the aggregated information.
        self.model.initialize(info)

    async def training_round(
            self,
            round_i: int,
        ) -> None:
        """Docstring."""
        self.logger.info("Initiating training round %s", round_i)
        clients = self._select_round_participants()
        await self._send_training_instructions(clients, round_i)
        self.logger.info("Awaiting clients' training results.")
        results = await self._collect_training_results(clients)
        self.logger.info("Conducting server-side optimization.")
        self._conduct_global_update(results)
        #revise: self._compute_global_metrics(results)

    def _select_round_participants(
            self,
        ) -> Set[str]:
        """Return the names of clients that should participate in the round."""
        return self.netwk.client_names

    async def _send_training_instructions(
            self,
            clients: Set[str],
            round_i: int,
        ) -> None:
        """Send training instructions to selected clients.

        Parameters
        ----------
        clients: set[str]
            Names of the clients participating in the training round.
        round_i: int
            Index of the training round.
        """
        # Set up shared training parameters.
        params = {
            "round_i": round_i,
            "weights": self.model.get_weights(),
            "batch_s": self.batch_size,
            "n_epoch": 1, # todo: add params (n_epoch, n_steps, timeout...)
        }  # type: Dict[str, Any]
        messages = {}  # type: Dict[str, messaging.Message]
        # Dispatch auxiliary variables (which may be client-specific).
        aux_var = self.optim.collect_aux_var()
        for client in clients:
            params["aux_var"] = {
                key: val[key].get(client, val[key])
                for key, val in aux_var.items()
            }
            messages[client] = messaging.TrainRequest(**params)
        # Send client-wise messages.
        await self.netwk.send_messages(messages)

    async def _collect_training_results(
            self,
            clients: Set[str],
        ) -> Dict[str, messaging.TrainReply]:
        """Collect training results for clients participating in a round.

        Parameters
        ----------
        clients: set[str]
            Names of the clients participating in the training round.

        Raises
        ------
        RuntimeError:
            If any client sent an incorrect message or reported
            failure to conduct the training step properly. Send
            CancelTraining to all clients before raising.

        Returns
        -------
        results: dict[str, TrainReply]
            Client-wise TrainReply message.
        """
        # Await clients' responses.
        replies = await self.netwk.wait_for_messages(clients)
        results = {}  # type: Dict[str, messaging.TrainReply]
        errors = {}  # type: Dict[str, str]
        for client, message in replies.items():
            if isinstance(message, messaging.TrainReply):
                results[client] = message
            elif isinstance(message, messaging.Error):
                errors[client] = f"Training failed: {message.message}"
            else:
                errors[client] = f"Unexpected message: {message}"
        # If any client has failed to send proper training results, raise.
        if errors:
            err_msg = "Training failed for another client."
            await self.netwk.send_messages({
                client: messaging.CancelTraining(errors.get(client, err_msg))
                for client in self.netwk.client_names
            })
            err_msg = f"Training failed for {len(errors)} clients:" + "".join(
                f"\n    {client}: {error}" for client, error in errors.items()
            )
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)
        # Otherwise, return collected results.
        return results

    def _conduct_global_update(
            self,
            results: Dict[str, messaging.TrainReply],
        ) -> None:
        """Use training results from clients to update the global model.

        Parameters
        ----------
        results: dict[str, TrainReply]
            Client-wise TrainReply message sent after a training round.
        """
        self.optim.process_aux_var(
            {client: result.aux_var for client, result in results.items()}
        )
        gradients = self.aggrg.aggregate(  # revise: pass n_epoch / t_spent / ?
            {client: result.updates for client, result in results.items()},
            {client: result.n_steps for client, result in results.items()}
        )
        self.optim.apply_gradients(self.model, gradients)

    def _compute_global_metrics(
            self,
            results: Dict[str, messaging.TrainReply],
        ) -> None:
        """Docstring."""
        # TODO: implement this
        raise NotImplementedError
