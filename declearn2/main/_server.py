# coding: utf-8

"""Server-side main Federated Learning orchestrating class."""

import asyncio
from typing import Any, Dict, Optional, Set, Type


from declearn2.communication import messaging
from declearn2.communication.api import Server
from declearn2.main.utils import (
    AggregationError,
    aggregate_clients_data_info,
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
        self._loss = {}  # type: Dict[int, float]

    def run(
            self,
            rounds: int,
            min_clients: int = 1,
            max_clients: Optional[int] = None,
            timeout: Optional[int] = None,
        ) -> None:
        """Docstring."""
        asyncio.run(self.training(rounds, min_clients, max_clients, timeout))

    async def training(
            self,
            rounds: int,
            min_clients: int,
            max_clients: Optional[int],
            timeout: Optional[int],
        ) -> None:
        """Orchestrate the federated training routine."""
        async with self.netwk:
            await self.initialization(min_clients, max_clients, timeout)
            round_i = 0
            while True:
                round_i += 1
                await self.training_round(round_i)
                await self.evaluation_round(round_i)
                if round_i >= rounds:
                    break
                # TODO: add early stopping criteria (based on former)
            self.logger.info("Stopping training.")
            await self.stop_training(round_i)

    async def initialization(
            self,
            min_clients: int,
            max_clients: Optional[int],
            timeout: Optional[int],
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
        self.logger.info("Starting clients registration process.")
        data_info = await self.netwk.wait_for_clients(
            min_clients, max_clients, timeout
        )
        self.logger.info("Clients' registration is now complete.")
        await self._process_data_info(data_info)
        # Serialize intialization information and send it to clients.
        message = messaging.InitRequest(
            model=self.model,
            optim=self.strat.build_client_optimizer(),
        )  # revise: strategy rather than optimizer?
        self.logger.info("Sending initialization requests to clients.")
        await self.netwk.broadcast_message(message)
        # Await a confirmation from clients that initialization went well.
        self.logger.info("Waiting for clients' responses.")
        replies = await self.netwk.wait_for_messages()
        self.logger.info("Received clients' responses.")
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
        self.logger.info("Initialization was successful.")

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
        clients = self._select_training_round_participants()
        await self._send_training_instructions(clients, round_i)
        self.logger.info("Awaiting clients' training results.")
        results = await self._collect_training_results(clients)
        self.logger.info("Conducting server-side optimization.")
        self._conduct_global_update(results)

    def _select_training_round_participants(
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
        return await self._collect_results(  # type: ignore
            clients, messaging.TrainReply, "training"
        )

    async def _collect_results(
            self,
            clients: Set[str],
            msgtype: Type[messaging.Message],
            context: str,
        ) -> Dict[str, messaging.Message]:
        """Collect some results sent by clients and ensure they are okay.

        Parameters
        ----------
        clients: set[str]
            Names of the clients that are expected to send messages.
        msgtype: type[messaging.Message]
            Type of message that clients are expected to send.
        context: str
            Context of the results collection (e.g. "training" or
            "evaluation"), used in logging or error messages.

        Raises
        ------
        RuntimeError:
            If any client sent an incorrect message or reported
            failure to conduct the evaluation step properly.
            Send CancelTraining to all clients before raising.

        Returns
        -------
        results: dict[str, `msgtype`]
            Client-wise collected messages.
        """
        # Await clients' responses and type-check them.
        replies = await self.netwk.wait_for_messages(clients)
        results = {}  # type: Dict[str, messaging.Message]
        errors = {}  # type: Dict[str, str]
        for client, message in replies.items():
            if isinstance(message, msgtype):
                results[client] = message
            elif isinstance(message, messaging.Error):
                errors[client] = f"{context} failed: {message.message}"
            else:
                errors[client] = f"Unexpected message: {message}"
        # If any client has failed to send proper results, raise.
        # future: modularize errors-handling behaviour
        if errors:
            err_msg = f"{context} failed for another client."
            await self.netwk.send_messages({
                client: messaging.CancelTraining(errors.get(client, err_msg))
                for client in self.netwk.client_names
            })
            err_msg = f"{context} failed for {len(errors)} clients:" + "".join(
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
        # Reformat received auxiliary variables and pass them to the Optimizer.
        aux_var = {}  # type: Dict[str, Dict[str, Dict[str, Any]]]
        for client, result in results.items():
            for module, params in result.aux_var.items():
                aux_var.setdefault(module, {})[client] = params
        self.optim.process_aux_var(aux_var)
        # Compute aggregated "gradients" (updates) and apply them to the model.
        gradients = self.aggrg.aggregate(  # revise: pass n_epoch / t_spent / ?
            {client: result.updates for client, result in results.items()},
            {client: result.n_steps for client, result in results.items()}
        )
        self.optim.apply_gradients(self.model, gradients)

    async def evaluation_round(
            self,
            round_i: int,
        ) -> None:
        """Docstring."""
        self.logger.info("Initiating evaluation round %s", round_i)
        clients = self._select_evaluation_round_participants()
        await self._send_evaluation_instructions(clients, round_i)
        self.logger.info("Awaiting clients' evaluation results.")
        results = await self._collect_evaluation_results(clients)
        self.logger.info("Aggregating evaluation results.")
        self._loss[round_i] = self._aggregate_evaluation_results(results)
        self.logger.info("Global loss is: %s", self._loss[round_i])

    def _select_evaluation_round_participants(
            self,
        ) -> Set[str]:
        """Return the names of clients that should participate in the round."""
        return self.netwk.client_names

    async def _send_evaluation_instructions(
            self,
            clients: Set[str],
            round_i: int,
        ) -> None:
        """Send evaluation instructions to selected clients.

        Parameters
        ----------
        clients: set[str]
            Names of the clients participating in the training round.
        round_i: int
            Index of the training round.
        """
        message = messaging.EvaluationRequest(
            round_i=round_i,
            weights=self.model.get_weights(),
            batch_s=self.batch_size,
        )
        await self.netwk.broadcast_message(message, clients)

    async def _collect_evaluation_results(
            self,
            clients: Set[str],
        ) -> Dict[str, messaging.EvaluationReply]:
        """Collect evaluation results for clients participating in a round.

        Parameters
        ----------
        clients: set[str]
            Names of the clients participating in the evaluation round.

        Raises
        ------
        RuntimeError:
            If any client sent an incorrect message or reported
            failure to conduct the evaluation step properly.
            Send CancelTraining to all clients before raising.

        Returns
        -------
        results: dict[str, EvaluationReply]
            Client-wise TrainReply message.
        """
        return await self._collect_results(  # type: ignore
            clients, messaging.EvaluationReply, "training"
        )

    def _aggregate_evaluation_results(
            self,
            results: Dict[str, messaging.EvaluationReply],
        ) -> float:
        """Aggregate evaluation results from clients into a global loss.

        Parameters
        ----------
        results: dict[str, EvaluationReply]
            Client-wise EvaluationReply message sent after
            an evaluation round.

        Returns
        -------
        loss: float
            The aggregated loss score computed from clients' ones.
        """
        total = 0.
        n_stp = 0
        for reply in results.values():  # future: enable re-weighting?
            total += reply.loss
            n_stp += reply.n_steps
        return total / n_stp

    async def stop_training(
            self,
            rounds: int
        ) -> None:
        """Notify clients that training is over and send final information.

        Parameters
        ----------
        rounds: int
            Number of training rounds taken until now.
        """
        message = messaging.StopTraining(
            weights=self.model.get_weights(),
            loss=self._loss[rounds],
            rounds=rounds
        )
        self.logger.info("Notifying clients that training is over.")
        await self.netwk.broadcast_message(message)
