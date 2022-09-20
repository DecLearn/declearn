# coding: utf-8

"""Server-side main Federated Learning orchestrating class."""

import asyncio
from typing import Any, Dict, Optional, Set, Tuple, Type, Union


from declearn.communication import NetworkServerConfig, messaging
from declearn.communication.api import Server
from declearn.main.utils import (
    AggregationError,
    Checkpointer,
    EarlyStopping,
    EvaluateConfig,
    RegisterConfig,
    TrainingConfig,
    aggregate_clients_data_info,
)
from declearn.model.api import Model
from declearn.strategy import Strategy
from declearn.utils import deserialize_object, get_logger


__all__ = [
    'FederatedServer',
]


class FederatedServer:
    """Server-side Federated Learning orchestrating class."""

    logger = get_logger("federated-server")

    def __init__(
            self,
            model: Union[Model, str, Dict[str, Any]],
            netwk: Union[Server, NetworkServerConfig, Dict[str, Any]],
            strategy: Strategy,  # future: revise Strategy, add config
            folder: Optional[str] = None,
        ) -> None:
        """Instantiate the orchestrating server for a federated learning task.

        Parameters
        ----------
        model: Model or dict or str
            Model instance, that may be serialized as an ObjectConfig,
            a config dict or a JSON file the path to which is provided.
        netwk: Server or NetworkServerConfig or dict
            Server communication endpoint instance, or configuration
            dict or dataclass enabling its instantiation.
        strategy: Strategy
            Strategy instance providing with instantiation methods for
            the server's updates-aggregator, the server-side optimizer
            and the clients-side one.
        folder: str or None, default=None
            Optional folder where to write out a model dump, round-
            wise weights checkpoints and global validation losses.
            If None, only record the loss metric and lowest-loss-
            yielding weights in memory (under `self.checkpoint`).
        """
        # Assign the wrapped Model.
        if not isinstance(model, Model):
            model = deserialize_object(model)  # type: ignore
        if not isinstance(model, Model):
            raise TypeError(
                "'model' should be a declearn Model, opt. in serialized form."
            )
        self.model = model
        # Assign the wrapped communication Server.
        if isinstance(netwk, dict):
            netwk = NetworkServerConfig(**netwk).build_server()
        elif isinstance(netwk, NetworkServerConfig):
            netwk = netwk.build_server()
        elif not isinstance(netwk, Server):
            raise TypeError(
                "'netwk' should be a declearn.communication.Server, "
                "or the valid configuration of one."
            )
        self.netwk = netwk
        # Assign the strategy and instantiate server-side objects.
        self.strat = strategy
        self.aggrg = self.strat.build_server_aggregator()
        self.optim = self.strat.build_server_optimizer()
        # Assign a model checkpointer.
        self.checkpointer = Checkpointer(self.model, folder)

    def _parse_config_dicts(
            self,
            regst_cfg: Union[RegisterConfig, Dict[str, Any], int],
            train_cfg: Union[TrainingConfig, Dict[str, Any]],
            valid_cfg: Union[EvaluateConfig, Dict[str, Any], None] = None,
        ) -> Tuple[RegisterConfig, TrainingConfig, EvaluateConfig]:
        """Parse input keyword arguments config dicts or dataclasses."""
        if isinstance(regst_cfg, int):
            regst_cfg = RegisterConfig(min_clients=regst_cfg)
        elif isinstance(regst_cfg, dict):
            regst_cfg = RegisterConfig(**regst_cfg)
        elif not isinstance(regst_cfg, RegisterConfig):
            raise TypeError(
                "'regst_cfg' should be a RegisterConfig instance or dict."
            )
        if isinstance(train_cfg, dict):
            train_cfg = TrainingConfig(**train_cfg)
        elif not isinstance(train_cfg, TrainingConfig):
            raise TypeError(
                "'train_cfg' should be a TrainingConfig instance or dict."
            )
        if valid_cfg is None:
            valid_cfg = EvaluateConfig(batch_size=train_cfg.batch_size)
        elif isinstance(valid_cfg, dict):
            valid_cfg = EvaluateConfig(**valid_cfg)
        elif not isinstance(valid_cfg, EvaluateConfig):
            raise TypeError(
                "'valid_cfg' should be a EvaluateConfig instance or dict."
            )
        return regst_cfg, train_cfg, valid_cfg

    def run(
            self,
            rounds: int,
            regst_cfg: Union[RegisterConfig, Dict[str, Any], int],
            train_cfg: Union[TrainingConfig, Dict[str, Any]],
            valid_cfg: Union[EvaluateConfig, Dict[str, Any], None] = None,
            early_stop: Optional[Union[EarlyStopping, Dict[str, Any]]] = None,
        ) -> None:
        """Orchestrate the federated learning routine.

        Parameters
        ----------
        rounds: int
            Maximum number of training rounds to perform.
        regst_cfg: RegisterConfig or dict or int
            Keyword arguments to specify clients-registration rules -
            formatted as a dict or a declearn.main.utils.RegisterConfig
            instance. Alternatively, use an int to specify the exact
            number of clients that are expected to register.
        train_cfg: TrainingConfig or dict
            Keyword arguments to specify effort constraints and data
            batching parameters for training rounds - formatted as a
            dict or a declearn.main.utils.TrainingConfig instance.
        valid_cfg: EvaluateConfig or dict or None, default=None
            Keyword arguments to specify effort constraints and data
            batching parameters for evaluation rounds. If None, use
            default arguments (1 epoch over batches of same size as
            for training, without shuffling nor samples dropping).
        early_stop: EarlyStopping or dict or None, default=None
            Optional EarlyStopping instance or configuration dict,
            specifying an early-stopping rule based on the global
            loss metric computed during evaluation rounds.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        configs = self._parse_config_dicts(regst_cfg, train_cfg, valid_cfg)
        if isinstance(early_stop, dict):
            early_stop = EarlyStopping(**early_stop)
        if not (isinstance(early_stop, EarlyStopping) or early_stop is None):
            raise TypeError(
                "'early_stop' must be None, int or EarlyStopping."
            )
        asyncio.run(self.async_run(rounds, configs, early_stop))

    async def async_run(
            self,
            rounds: int,
            configs: Tuple[RegisterConfig, TrainingConfig, EvaluateConfig],
            early_stop: Optional[EarlyStopping],
        ) -> None:
        """Orchestrate the federated learning routine.

        Note: this method is the async backend of `self.run`.

        Parameters
        ----------
        rounds: int
            Maximum number of training rounds to perform.
        configs: (RegisterConfig, TrainingConfig, EvaluateConfig) tuple
            Dataclass instances wrapping hyper-parameters specifying
            clients-registration and training and evaluation rounds.
        early_stop: EarlyStopping or None
            Optional EarlyStopping instance adding a stopping criterion
            based on the global-evaluation-loss's evolution over rounds.
        """
        regst_cfg, train_cfg, valid_cfg = configs
        async with self.netwk:
            await self.initialization(regst_cfg)
            self.checkpointer.save_model()
            self.checkpointer.checkpoint(float("inf"))  # save initial weights
            round_i = 0
            while True:
                round_i += 1
                await self.training_round(round_i, train_cfg)
                await self.evaluation_round(round_i, valid_cfg)
                if not self._keep_training(round_i, rounds, early_stop):
                    break
            self.logger.info("Stopping training.")
            await self.stop_training(round_i)

    async def initialization(
            self,
            regst_cfg: RegisterConfig,
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
            regst_cfg.min_clients, regst_cfg.max_clients, regst_cfg.timeout
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
            train_cfg: TrainingConfig,
        ) -> None:
        """Orchestrate a training round.

        Parameters
        ----------
        round_i: int
            Index of the training round.
        train_cfg: TrainingConfig
            TrainingConfig dataclass instance wrapping data-batching
            and computational effort constraints hyper-parameters.
        """
        self.logger.info("Initiating training round %s", round_i)
        clients = self._select_training_round_participants()
        await self._send_training_instructions(clients, round_i, train_cfg)
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
            train_cfg: TrainingConfig,
        ) -> None:
        """Send training instructions to selected clients.

        Parameters
        ----------
        clients: set[str]
            Names of the clients participating in the training round.
        round_i: int
            Index of the training round.
        train_cfg: TrainingConfig
            TrainingConfig dataclass instance wrapping data-batching
            and computational effort constraints hyper-parameters.
        """
        # Set up shared training parameters.
        params = {
            "round_i": round_i,
            "weights": self.model.get_weights(),
            "batches": train_cfg.batch_cfg,
            "n_epoch": train_cfg.n_epoch,
            "n_steps": train_cfg.n_steps,
            "timeout": train_cfg.timeout,
        }  # type: Dict[str, Any]
        messages = {}  # type: Dict[str, messaging.Message]
        # Dispatch auxiliary variables (which may be client-specific).
        aux_var = self.optim.collect_aux_var()
        for client in clients:
            params["aux_var"] = {
                key: val.get(client, val)
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
            valid_cfg: EvaluateConfig,
        ) -> None:
        """Orchestrate an evaluation round.

        Parameters
        ----------
        round_i: int
            Index of the evaluation round.
        valid_cfg: EvaluateConfig
            EvaluateConfig dataclass instance wrapping data-batching
            and computational effort constraints hyper-parameters.
        """
        self.logger.info("Initiating evaluation round %s", round_i)
        clients = self._select_evaluation_round_participants()
        await self._send_evaluation_instructions(clients, round_i, valid_cfg)
        self.logger.info("Awaiting clients' evaluation results.")
        results = await self._collect_evaluation_results(clients)
        self.logger.info("Aggregating evaluation results.")
        loss = self._aggregate_evaluation_results(results)
        self.logger.info("Global loss is: %s", loss)
        self.checkpointer.checkpoint(loss)

    def _select_evaluation_round_participants(
            self,
        ) -> Set[str]:
        """Return the names of clients that should participate in the round."""
        return self.netwk.client_names

    async def _send_evaluation_instructions(
            self,
            clients: Set[str],
            round_i: int,
            valid_cfg: EvaluateConfig,
        ) -> None:
        """Send evaluation instructions to selected clients.

        Parameters
        ----------
        clients: set[str]
            Names of the clients participating in the evaluation round.
        round_i: int
            Index of the evaluation round.
        valid_cfg: EvaluateConfig
            EvaluateConfig dataclass instance wrapping data-batching
            and computational effort constraints hyper-parameters.
        """
        message = messaging.EvaluationRequest(
            round_i=round_i,
            weights=self.model.get_weights(),
            batches=valid_cfg.batch_cfg,
            # future: n_steps=self.valid_cfg.n_steps,
            # future: timeout=self.valid_cfg.timeout,
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
            total += reply.loss * reply.n_steps
            n_stp += reply.n_steps
        return total / n_stp

    def _keep_training(
            self,
            round_i: int,
            rounds: int,
            early_stop: Optional[EarlyStopping],
        ) -> bool:
        """Decide whether training should continue.

        Parameters
        ----------
        round_i: int
            Index of the latest achieved training round.
        rounds: int
            Maximum number of rounds that are planned.
        early_stop: EarlyStopping or None
            Optional EarlyStopping instance adding a stopping criterion
            based on the global-evaluation-loss's evolution over rounds.
        """
        if round_i >= rounds:
            self.logger.info("Maximum number of training rounds reached.")
            return False
        if early_stop is not None:
            early_stop.update(self.checkpointer.get_loss(round_i))
            if not early_stop.keep_training:
                self.logger.info("Early stopping criterion reached.")
                return False
        return True

    async def stop_training(
            self,
            rounds: int,
        ) -> None:
        """Notify clients that training is over and send final information.

        Parameters
        ----------
        rounds: int
            Number of training rounds taken until now.
        """
        self.checkpointer.reset_best_weights()
        message = messaging.StopTraining(
            weights=self.model.get_weights(),
            loss=min(self.checkpointer.get_loss(i) for i in range(rounds)),
            rounds=rounds,
        )
        self.logger.info("Notifying clients that training is over.")
        await self.netwk.broadcast_message(message)