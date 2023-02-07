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

"""Server-side main Federated Learning orchestrating class."""

import asyncio
import dataclasses
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np

from declearn.communication import NetworkServerConfig, messaging
from declearn.communication.api import NetworkServer
from declearn.main.config import (
    EvaluateConfig,
    FLOptimConfig,
    FLRunConfig,
    TrainingConfig,
)
from declearn.main.utils import (
    AggregationError,
    Checkpointer,
    EarlyStopping,
    aggregate_clients_data_info,
)
from declearn.metrics import MetricInputType, MetricSet
from declearn.model.api import Model, Vector
from declearn.utils import deserialize_object, get_logger


__all__ = [
    "FederatedServer",
]


MessageT = TypeVar("MessageT", bound=messaging.Message)


class FederatedServer:
    """Server-side Federated Learning orchestrating class."""

    # one-too-many attribute; pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        model: Union[Model, str, Dict[str, Any]],
        netwk: Union[NetworkServer, NetworkServerConfig, Dict[str, Any], str],
        optim: Union[FLOptimConfig, str, Dict[str, Any]],
        metrics: Union[MetricSet, List[MetricInputType], None] = None,
        checkpoint: Union[Checkpointer, Dict[str, Any], str, None] = None,
        logger: Union[logging.Logger, str, None] = None,
    ) -> None:
        """Instantiate the orchestrating server for a federated learning task.

        Parameters
        ----------
        model: Model or dict or str
            Model instance, that may be serialized as an ObjectConfig,
            a config dict or a JSON file the path to which is provided.
        netwk: NetworkServer or NetworkServerConfig or dict or str
            NetworkServer communication endpoint instance, or configuration
            dict, dataclass or path to a TOML file enabling its instantiation.
            In the latter three cases, the object's default logger will
            be set to that of this `FederatedServer`.
        optim: FLOptimConfig or dict or str
            FLOptimConfig instance or instantiation dict (using
            the `from_params` method) or TOML configuration file path.
            This object specifies the optimizers to use by the clients
            and the server, as well as the client-updates aggregator.
        metrics: MetricSet or list[MetricInputType] or None, default=None
            MetricSet instance or list of Metric instances and/or specs
            to wrap into one, defining evaluation metrics to compute in
            addition to the model's loss.
            If None, only compute and report the model's loss.
        checkpoint: Checkpointer or dict or str or None, default=None
            Optional Checkpointer instance or instantiation dict to be
            used so as to save round-wise model, optimizer and metrics.
            If a single string is provided, treat it as the checkpoint
            folder path and use default values for other parameters.
        logger: logging.Logger or str or None, default=None,
            Logger to use, or name of a logger to set up with
            `declearn.utils.get_logger`. If None, use `type(self)`.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        # Assign the logger.
        if not isinstance(logger, logging.Logger):
            logger = get_logger(logger or type(self).__name__)
        self.logger = logger
        # Assign the wrapped Model.
        if not isinstance(model, Model):
            model = deserialize_object(model)  # type: ignore
        if not isinstance(model, Model):
            raise TypeError(
                "'model' should be a declearn Model, opt. in serialized form."
            )
        self.model = model
        # Assign the wrapped NetworkServer.
        if isinstance(netwk, str):
            netwk = NetworkServerConfig.from_toml(netwk)
        elif isinstance(netwk, dict):
            netwk = NetworkServerConfig(**netwk)
        if isinstance(netwk, NetworkServerConfig):
            if netwk.logger is None:
                netwk.logger = self.logger
            netwk = netwk.build_server()
        if not isinstance(netwk, NetworkServer):
            raise TypeError(
                "'netwk' should be a declearn.communication.api.NetworkServer,"
                " or the valid configuration of one."
            )
        self.netwk = netwk
        # Assign the wrapped FLOptimConfig.
        if isinstance(optim, str):
            optim = FLOptimConfig.from_toml(optim)
        elif isinstance(optim, dict):
            optim = FLOptimConfig.from_params(**optim)
        if not isinstance(optim, FLOptimConfig):
            raise TypeError(
                "'optim' should be a declearn.main.config.FLOptimConfig "
                "or a dict of parameters or the path to a TOML file from "
                "which to instantiate one."
            )
        self.aggrg = optim.aggregator
        self.optim = optim.server_opt
        self.c_opt = optim.client_opt
        # Assign the wrapped MetricSet.
        self.metrics = MetricSet.from_specs(metrics)
        # Assign an optional checkpointer.
        if checkpoint is not None:
            checkpoint = Checkpointer.from_specs(checkpoint)
        self.ckptr = checkpoint
        # Set up private attributes to record the loss values and best weights.
        self._loss = {}  # type: Dict[int, float]
        self._best = None  # type: Optional[Vector]

    def run(
        self,
        config: Union[FLRunConfig, str, Dict[str, Any]],
    ) -> None:
        """Orchestrate the federated learning routine.

        Parameters
        ----------
        config: FLRunConfig or str or dict
            Container instance wrapping grouped hyper-parameters that
            specify the federated learning process, including clients
            registration, training and validation rounds' setup, plus
            an optional early-stopping criterion.
            May be a str pointing to a TOML configuration file.
            May be as a dict of keyword arguments to be parsed.
        """
        if isinstance(config, dict):
            config = FLRunConfig.from_params(**config)
        if isinstance(config, str):
            config = FLRunConfig.from_toml(config)
        if not isinstance(config, FLRunConfig):
            raise TypeError("'config' should be a FLRunConfig object or str.")
        asyncio.run(self.async_run(config))

    async def async_run(
        self,
        config: FLRunConfig,
    ) -> None:
        """Orchestrate the federated learning routine.

        Note: this method is the async backend of `self.run`.

        Parameters
        ----------
        config: FLRunConfig
            Container instance wrapping grouped hyper-parameters that
            specify the federated learning process, including clients
            registration, training and validation rounds' setup, plus
            optional elements: local differential-privacy parameters,
            and/or an early-stopping criterion.
        """
        # Instantiate the early-stopping criterion, if any.
        early_stop = None  # type: Optional[EarlyStopping]
        if config.early_stop is not None:
            early_stop = config.early_stop.instantiate()
        # Start the communications server and run the FL process.
        async with self.netwk:
            # Conduct the initialization phase.
            await self.initialization(config)
            if self.ckptr:
                self.ckptr.checkpoint(self.model, self.optim, first_call=True)
            # Iteratively run training and evaluation rounds.
            round_i = 0
            while True:
                round_i += 1
                await self.training_round(round_i, config.training)
                await self.evaluation_round(round_i, config.evaluate)
                if not self._keep_training(round_i, config.rounds, early_stop):
                    break
            # Interrupt training when time comes.
            self.logger.info("Stopping training.")
            await self.stop_training(round_i)

    async def initialization(
        self,
        config: FLRunConfig,
    ) -> None:
        """Orchestrate the initialization steps to set up training.

        Wait for clients to register and process their data information.
        Send instructions to clients to set up their model and optimizer.
        Await clients to have finalized their initialization step; raise
        and cancel training if issues are reported back.

        Parameters
        ----------
        config: FLRunConfig
            Container instance wrapping hyper-parameters that specify
            the planned federated learning process, including clients
            registration ones as a RegisterConfig dataclass instance.

        Raises
        ------
        RuntimeError:
            In case any of the clients returned an Error message rather
            than an Empty ping-back message. Send CancelTraining to all
            clients before raising.
        """
        # Gather the RegisterConfig instance from the main FLRunConfig.
        regst_cfg = config.register
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
            optim=self.c_opt,
            metrics=self.metrics.get_config()["metrics"],
            dpsgd=config.privacy is not None,
        )
        self.logger.info("Sending initialization requests to clients.")
        await self.netwk.broadcast_message(message)
        # Await a confirmation from clients that initialization went well.
        # If any client has failed to initialize, raise.
        self.logger.info("Waiting for clients' responses.")
        await self._collect_results(
            clients=self.netwk.client_names,
            msgtype=messaging.GenericMessage,
            context="initialization",
        )
        # If local differential privacy is configured, set it up.
        if config.privacy is not None:
            await self._initialize_dpsgd(config)
        self.logger.info("Initialization was successful.")

    async def _process_data_info(
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
            messages = {
                client: messaging.CancelTraining(reason)
                for client, reason in exc.messages.items()
            }  # type: Dict[str, messaging.Message]
            await self.netwk.send_messages(messages)
            self.logger.error(exc.error)
            raise exc
        # Otherwise, initialize the model based on the aggregated information.
        self.model.initialize(info)

    async def _collect_results(
        self,
        clients: Set[str],
        msgtype: Type[MessageT],
        context: str,
    ) -> Dict[str, MessageT]:
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
        results = {}  # type: Dict[str, MessageT]
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
            messages = {
                client: messaging.CancelTraining(errors.get(client, err_msg))
                for client in self.netwk.client_names
            }  # type: Dict[str, messaging.Message]
            await self.netwk.send_messages(messages)
            err_msg = f"{context} failed for {len(errors)} clients:" + "".join(
                f"\n    {client}: {error}" for client, error in errors.items()
            )
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)
        # Otherwise, return collected results.
        return results

    async def _initialize_dpsgd(
        self,
        config: FLRunConfig,
    ) -> None:
        """Send a differential privacy setup request to all registered clients.

        Parameters
        ----------
        config: FLRunConfig
            FLRunConfig wrapping information on the overall FL process
            and on the local DP parameters. Its `privacy` section must
            be defined.
        """
        self.logger.info("Sending privacy requests to all clients.")
        assert config.privacy is not None  # else this method is not called
        params = {
            "rounds": config.rounds,
            "batches": config.training.batch_cfg,
            "n_epoch": config.training.n_epoch,
            "n_steps": config.training.n_steps,
            **dataclasses.asdict(config.privacy),
        }  # type: Dict[str, Any]
        message = messaging.PrivacyRequest(**params)
        await self.netwk.broadcast_message(message)
        self.logger.info("Waiting for clients' responses.")
        await self._collect_results(
            clients=self.netwk.client_names,
            msgtype=messaging.GenericMessage,
            context="Privacy initialization",
        )
        self.logger.info("Privacy requests were processed by clients.")

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
        results = await self._collect_results(
            clients, messaging.TrainReply, "training"
        )
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
            **train_cfg.message_params,
        }  # type: Dict[str, Any]
        messages = {}  # type: Dict[str, messaging.Message]
        # Dispatch auxiliary variables (which may be client-specific).
        aux_var = self.optim.collect_aux_var()
        for client in clients:
            params["aux_var"] = {
                key: val.get(client, val) for key, val in aux_var.items()
            }
            messages[client] = messaging.TrainRequest(**params)
        # Send client-wise messages.
        await self.netwk.send_messages(messages)

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
        # revise: pass n_epoch / t_spent / ?
        gradients = self.aggrg.aggregate(
            {client: result.updates for client, result in results.items()},
            {client: result.n_steps for client, result in results.items()},
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
        # Send evaluation requests and collect clients' replies.
        self.logger.info("Initiating evaluation round %s", round_i)
        clients = self._select_evaluation_round_participants()
        await self._send_evaluation_instructions(clients, round_i, valid_cfg)
        self.logger.info("Awaiting clients' evaluation results.")
        results = await self._collect_results(
            clients, messaging.EvaluationReply, "evaluation"
        )
        # Compute and report aggregated evaluation metrics.
        self.logger.info("Aggregating evaluation results.")
        loss, metrics = self._aggregate_evaluation_results(results)
        self.logger.info("Averaged loss is: %s", loss)
        if metrics:
            self.logger.info(
                "Other averaged scalar metrics are: %s",
                {k: v for k, v in metrics.items() if isinstance(v, float)},
            )
        # Optionally checkpoint the model, optimizer and metrics.
        if self.ckptr:
            self._checkpoint_after_evaluation(metrics, results)
        # Record the global loss, and update the kept "best" weights.
        self._loss[round_i] = loss
        if loss == min(self._loss.values()):
            self._best = self.model.get_weights()

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
            **valid_cfg.message_params,
        )
        await self.netwk.broadcast_message(message, clients)

    def _aggregate_evaluation_results(
        self,
        results: Dict[str, messaging.EvaluationReply],
    ) -> Tuple[float, Dict[str, Union[float, np.ndarray]]]:
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
        metrics: dict[str, (float | np.ndarray)]
            The aggregated evaluation metrics computes from clients' ones.
        """
        # Reset the local MetricSet and set up ad hoc variables for the loss.
        loss = 0.0
        dvsr = 0.0
        self.metrics.reset()
        # Iteratively update the MetricSet and loss floats based on results.
        for client, reply in results.items():
            # Case when the client reported some metrics.
            if reply.metrics:
                states = reply.metrics.copy()
                # Update the global metrics based on the local ones.
                s_loss = states.pop("loss")
                loss += s_loss["current"]  # type: ignore
                dvsr += s_loss["divisor"]  # type: ignore
                self.metrics.agg_states(states)
            # Case when the client only reported the aggregated local loss.
            else:
                self.logger.info(
                    "Client %s refused to share their local metrics.", client
                )
                loss += reply.loss
                dvsr += reply.n_steps
        # Compute the final results.
        metrics = self.metrics.get_result()
        loss = loss / dvsr
        return loss, metrics

    def _checkpoint_after_evaluation(
        self,
        metrics: Dict[str, Union[float, np.ndarray]],
        results: Dict[str, messaging.EvaluationReply],
    ) -> None:
        """Checkpoint the current model, optimizer and evaluation metrics.

        This method is meant to be called at the end of an evaluation round.

        Parameters
        ----------
        metrics: dict[str, (float|np.ndarray)]
            Aggregated evaluation metrics to checkpoint.
        results: dict[str, EvaluationReply]
            Client-wise EvaluationReply messages, based on which
            `metrics` were already computed.
        """
        # This method only works when a checkpointer is used.
        if self.ckptr is None:
            raise RuntimeError(
                "`_checkpoint_after_evaluation` was called without "
                "the FederatedServer having a Checkpointer."
            )
        # Checkpoint the model, optimizer and global evaluation metrics.
        timestamp = self.ckptr.checkpoint(
            model=self.model, optimizer=self.optim, metrics=metrics
        )
        # Checkpoint the client-wise metrics (or at least their loss).
        # Use the same timestamp label as for global metrics and states.
        local = MetricSet.from_config(self.metrics.get_config())
        for client, reply in results.items():
            if reply.metrics:
                local.reset()
                local.agg_states(reply.metrics)
                metrics = local.get_result()
            else:
                metrics = {"loss": reply.loss}
            self.ckptr.save_metrics(
                metrics=local.get_result(),
                prefix=f"metrics_{client}",
                append=bool(self._loss),
                timestamp=timestamp,
            )

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
            early_stop.update(self._loss[round_i])
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
        self.logger.info("Recovering weights that yielded the lowest loss.")
        message = messaging.StopTraining(
            weights=self._best or self.model.get_weights(),
            loss=min(self._loss.values()) if self._loss else float("nan"),
            rounds=rounds,
        )
        self.logger.info("Notifying clients that training is over.")
        await self.netwk.broadcast_message(message)
        if self.ckptr:
            path = f"{self.ckptr.folder}/model_state_best.json"
            self.logger.info("Checkpointing final weights under %s.", path)
            self.model.set_weights(message.weights)
            self.ckptr.save_model(self.model, timestamp="best")
