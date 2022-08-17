# coding: utf-8

"""Server-side main Federated Learning orchestrating class."""

import dataclasses
import functools
from typing import Any, Dict, List, Optional


from declearn2.communication.api import Server
from declearn2.main._data_info import (
    AggregationError, aggregate_clients_data_info
)
from declearn2.model.api import Model, Vector
from declearn2.strategy import Strategy
from declearn2.utils import get_logger, serialize_object


INITIALIZE = "initialize from sent params"
CANCEL_TRAINING = "training is cancelled"
TRAIN_ONE_ROUND = "train for one round"


@dataclasses.dataclass
class TrainingResults:
    """Dataclass to store training results sent by clients to the server."""

    updates: Vector
    aux_var: Dict[str, Any]
    n_steps: int
    error: Optional[str] = None
    #n_epoch/time
    #loss(es)
    #scores


class FederatedServer:
    """Server-side Federated Learning orchestrating class."""

    logger = get_logger("federated-server")

    def __init__(
            self,
            model: Model,    # revise: from_config
            server: Server,  # revise: from_config
            strategy: Strategy,  # revise: from_config
        ) -> None:
        """Docstring."""
        self.model = model
        self.netwk = server
        self.strat = strategy
        self.aggrg = self.strat.build_server_aggregator()
        self.optim = self.strat.build_server_optimizer()

    def run(
            self,
            rounds: int,
        ) -> None:
        """Docstring."""
        task = functools.partial(self.training, rounds=rounds)
        self.netwk.run_until_complete(task)

    async def training(
            self,
            rounds: int,
        ) -> None:
        """Docstring."""
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
        """Docstring."""
        # Wait for clients to register and process their data information.
        data_info = await self.netwk.wait_for_clients()  # revise: nb_clients
        await self._process_data_info(data_info)
        # Serialize intialization information and send it to clients.
        params = {  # revise: strategy rather than optimizer?
            "model": serialize_object(self.model),
            "optim": serialize_object(self.strat.build_client_optimizer()),
        }
        self.netwk.broadcast_message(INITIALIZE, params)
        # Await a confirmation from clients that initialization went well.
        replies = await self.netwk.wait_for_messages()
        errors = {
            client: msg["error"]
            for client, msg in replies.items()
            if msg.get("error") is not None
        }
        # If any client has failed to send proper training results, raise.
        if errors:
            message = "Initialization failed for another client."
            for client in self.netwk.client_names:
                await self.netwk.send_message(
                    client, CANCEL_TRAINING,
                    {"reason": errors.get(client, message)}
                )
            message = f"Initialization failed for {len(errors)} clients:"
            message += "".join(
                f"\n    {client}: {error}" for client, error in errors.items()
            )
            self.logger.error(message)
            raise RuntimeError(message)

    async def _process_data_info(  # revise: drop async
            self,
            clients_data_info: Dict[str, Dict[str, Any]],
        ) -> None:
        """Docstring."""
        fields = self.model.required_data_info  # revise: add optimizer, etc.
        # Try aggregating the input data_info.
        try:
            info = aggregate_clients_data_info(clients_data_info, fields)
        # In case of failure, cancel training, notify clients, log and raise.
        except AggregationError as exc:
            for client, reason in exc.messages.items():
                await self.netwk.send_message(
                    client, CANCEL_TRAINING, {"reason": reason}
                )
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
        results = await self._collect_training_results()#clients)
        self.logger.info("Conducting server-side optimization.")
        self._conduct_global_update(results)
        #revise: self._compute_global_metrics(results)

    def _select_round_participants(
            self,
        ) -> List[str]:
        """Return the list of clients that should participate in the round."""
        return list(self.netwk.client_names)

    async def _send_training_instructions(
            self,
            clients: List[str],
            round_i: int,
        ) -> None:
        """Send training instructions to selected clients."""
        aux_var = self.optim.collect_aux_var()
        params = {
            "round": round_i,
            "weights": self.model.get_weights(),
            "batch_size": self.strat.batch_size,  # todo: implement/revise
            # todo: add params (n_epochs, n_steps, timeout...)
        }  # type: Dict[str, Any]
        for client in clients:
            params["aux_var"] = {
                key: val[key].get(client, val[key])
                for key, val in aux_var.items()
            }
            await self.netwk.send_message(client, TRAIN_ONE_ROUND, params)

    async def _collect_training_results(
            self,
            #clients: List[str],
        ) -> Dict[str, TrainingResults]:
        """Collect training results for clients participating in a round.

        Raise a RuntimeError if any client sent an incorrect message
        or reported a failure to conduct the training step properly,
        after sending cancelling instructions to all clients.

        Return a {client_name: TrainingResults} dict otherwise.
        """
        # Await clients' responses.
        replies = await self.netwk.wait_for_messages()  # revise: specify clients
        results = {}  # type: Dict[str, TrainingResults]
        errors = {}  # type: Dict[str, str]
        for client, params in replies.items():  # revise: return Messages?
            try:
                results[client] = TrainingResults(**params)
            except TypeError as exception:
                errors[client] = repr(exception)
            else:  # revise: modularize failures' handling
                if results[client].error:
                    errors[client] = (
                        f"Training failed: {results[client].error}."
                    )
        # If any client has failed to send proper training results, raise.
        if errors:
            message = "Training failed for another client."
            for client in self.netwk.client_names:
                await self.netwk.send_message(
                    client, CANCEL_TRAINING,
                    {"reason": errors.get(client, message)}
                )
            message = f"Training failed for {len(errors)} clients:" + "".join(
                f"\n    {client}: {error}" for client, error in errors.items()
            )
            self.logger.error(message)
            raise RuntimeError(message)
        # Otherwise, return collected results.
        return results

    def _conduct_global_update(
            self,
            results: Dict[str, TrainingResults],
        ) -> None:
        """Use training results from clients to update the global model."""
        self.optim.process_aux_var(
            {client: result.aux_var for client, result in results.items()}
        )
        gradients = self.aggrg.aggregate(
            {client: result.updates for client, result in results.items()},
            {client: result.n_steps for client, result in results.items()}
        )
        self.optim.apply_gradients(self.model, gradients)

    def _compute_global_metrics(
            self,
            results: Dict[str, TrainingResults],
        ) -> None:
        """Docstring."""
        # TODO: implement this
        raise NotImplementedError
