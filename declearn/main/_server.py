# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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
import copy
import dataclasses
import logging
from typing import (  # fmt: off
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from declearn import messaging
from declearn.communication import NetworkServerConfig
from declearn.communication.api import NetworkServer
from declearn.main.config import (
    EvaluateConfig,
    FairnessConfig,
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
from declearn.metrics._mean import MeanState
from declearn.model.api import Model, Vector
from declearn.optimizer.modules import AuxVar
from declearn.secagg import messaging as secagg_messaging
from declearn.secagg import parse_secagg_config_server
from declearn.secagg.api import Decrypter, SecaggConfigServer
from declearn.utils import deserialize_object, get_logger

__all__ = [
    "FederatedServer",
]


MessageT = TypeVar("MessageT", bound=messaging.Message)


class FederatedServer:
    """Server-side Federated Learning orchestrating class."""

    # one-too-many attribute; pylint: disable=too-many-instance-attributes
    # pylint: disable-next=too-many-positional-arguments
    def __init__(  # noqa: PLR0913
        self,
        model: Union[Model, str, Dict[str, Any]],
        netwk: Union[NetworkServer, NetworkServerConfig, Dict[str, Any], str],
        optim: Union[FLOptimConfig, str, Dict[str, Any]],
        metrics: Union[MetricSet, List[MetricInputType], None] = None,
        secagg: Union[SecaggConfigServer, Dict[str, Any], None] = None,
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
        secagg: SecaggConfigServer or dict or None, default=None
            Optional SecAgg config and setup controller
            or dict of kwargs to set one up.
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
        self.model = self._parse_model(model)
        # Assign the wrapped NetworkServer.
        self.netwk = self._parse_netwk(netwk, logger=self.logger)
        # Assign the wrapped FLOptimConfig.
        optim = self._parse_optim(optim)
        self.aggrg = optim.aggregator
        self.optim = optim.server_opt
        self.c_opt = optim.client_opt
        self.fairness = optim.fairness  # note: optional
        # Assign the wrapped MetricSet.
        self.metrics = MetricSet.from_specs(metrics)
        # Assign an optional checkpointer.
        if checkpoint is not None:
            checkpoint = Checkpointer.from_specs(checkpoint)
        self.ckptr = checkpoint
        # Assign the optional SecAgg config and declare a Decrypter slot.
        self.secagg = self._parse_secagg(secagg)
        self._decrypter: Optional[Decrypter] = None
        self._secagg_peers: Set[str] = set()
        # Set up private attributes to record the loss values and best weights.
        self._losses: List[float] = []
        self._best: Optional[Vector] = None
        # Set up a private attribute to prevent redundant weights sharing.
        self._clients_holding_latest_model: Set[str] = set()

    @staticmethod
    def _parse_model(
        model: Union[Model, str, Dict[str, Any]],
    ) -> Model:
        """Parse 'model' instantiation argument."""
        if isinstance(model, Model):
            return model
        if isinstance(model, (str, dict)):
            try:
                output = deserialize_object(model)  # type: ignore[arg-type]
            except Exception as exc:
                raise TypeError(
                    "'model' input deserialization failed."
                ) from exc
            if isinstance(output, Model):
                return output
            raise TypeError(
                f"'model' input was deserialized into '{type(output)}', "
                "whereas a declearn 'Model' instance was expected."
            )
        raise TypeError(
            "'model' should be a declearn Model, optionally in serialized "
            f"form, not '{type(model)}'"
        )

    @staticmethod
    def _parse_netwk(
        netwk: Union[NetworkServer, NetworkServerConfig, Dict[str, Any], str],
        logger: logging.Logger,
    ) -> NetworkServer:
        """Parse 'netwk' instantiation argument."""
        # Case when a NetworkServer instance is provided: return.
        if isinstance(netwk, NetworkServer):
            return netwk
        # Case when a NetworkServerConfig is expected: verify or parse.
        if isinstance(netwk, NetworkServerConfig):
            config = netwk
        elif isinstance(netwk, str):
            config = NetworkServerConfig.from_toml(netwk)
        elif isinstance(netwk, dict):
            config = NetworkServerConfig(**netwk)
        else:
            raise TypeError(
                "'netwk' should be a 'NetworkServer' instance or the valid "
                f"configuration of one, not '{type(netwk)}'."
            )
        # Instantiate from the (parsed) config.
        if config.logger is None:
            config.logger = logger
        return config.build_server()

    @staticmethod
    def _parse_optim(
        optim: Union[FLOptimConfig, str, Dict[str, Any]],
    ) -> FLOptimConfig:
        """Parse 'optim' instantiation argument."""
        if isinstance(optim, FLOptimConfig):
            return optim
        if isinstance(optim, str):
            return FLOptimConfig.from_toml(optim)
        if isinstance(optim, dict):
            return FLOptimConfig.from_params(**optim)
        raise TypeError(
            "'optim' should be a declearn.main.config.FLOptimConfig "
            "or a dict of parameters or the path to a TOML file from "
            f"which to instantiate one, not '{type(optim)}'."
        )

    @staticmethod
    def _parse_secagg(
        secagg: Union[SecaggConfigServer, Dict[str, Any], None],
    ) -> Optional[SecaggConfigServer]:
        """Parse 'secagg' instantiation argument."""
        if secagg is None:
            return None
        if isinstance(secagg, SecaggConfigServer):
            return secagg
        if isinstance(secagg, dict):
            try:
                return parse_secagg_config_server(**secagg)
            except Exception as exc:
                raise TypeError("Failed to parse 'secagg' inputs.") from exc
        raise TypeError(
            "'secagg' should be a 'SecaggConfigServer' instance or a dict "
            f"of keyword arguments to set one up, not '{type(secagg)}'."
        )

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
            fairness evaluation parameters, and/or an early-stopping
            criterion.
        """
        # Instantiate the early-stopping criterion, if any.
        early_stop: Optional[EarlyStopping] = None
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
                await self.fairness_round(round_i, config.fairness)
                round_i += 1
                await self.training_round(round_i, config.training)
                await self.evaluation_round(round_i, config.evaluate)
                # Decide whether to keep training for at least one round.
                if not self._keep_training(round_i, config.rounds, early_stop):
                    break
            # When checkpointing, force evaluating the last model.
            if self.ckptr is not None:
                if round_i % config.evaluate.frequency:
                    await self.evaluation_round(
                        round_i, config.evaluate, force_run=True
                    )
                await self.fairness_round(
                    round_i, config.fairness, force_run=True
                )
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
        RuntimeError
            In case any of the clients returned an Error message rather
            than an Empty ping-back message. Send CancelTraining to all
            clients before raising.
        """
        # Gather the RegisterConfig instance from the main FLRunConfig.
        regst_cfg = config.register
        # Wait for clients to register.
        self.logger.info("Starting clients registration process.")
        await self.netwk.wait_for_clients(
            regst_cfg.min_clients, regst_cfg.max_clients, regst_cfg.timeout
        )
        self.logger.info("Clients' registration is now complete.")
        # When needed, prompt clients for metadata and process them.
        await self._require_and_process_data_info()
        # Serialize intialization information and send it to clients.
        message = messaging.InitRequest(
            model=self.model,
            optim=self.c_opt,
            aggrg=self.aggrg,
            metrics=self.metrics.get_config()["metrics"],
            dpsgd=config.privacy is not None,
            secagg=None if self.secagg is None else self.secagg.secagg_type,
            fairness=self.fairness is not None,
        )
        self.logger.info("Sending initialization requests to clients.")
        await self.netwk.broadcast_message(message)
        # Await a confirmation from clients that initialization went well.
        # If any client has failed to initialize, raise.
        self.logger.info("Waiting for clients' responses.")
        await self._collect_results(
            clients=self.netwk.client_names,
            msgtype=messaging.InitReply,
            context="Initialization",
        )
        # If local differential privacy is configured, set it up.
        if config.privacy is not None:
            await self._initialize_dpsgd(config)
        # If fairness-aware federated learning is configured, set it up.
        if self.fairness is not None:
            # When SecAgg is to be used, setup controllers first.
            if self.secagg is not None:
                await self.setup_secagg()
            # Call the setup routine of the held fairness controller.
            self.aggrg = await self.fairness.setup_fairness(
                netwk=self.netwk, aggregator=self.aggrg, secagg=self._decrypter
            )
        self.logger.info("Initialization was successful.")

    async def _require_and_process_data_info(
        self,
    ) -> None:
        """Collect, validate, aggregate and make use of clients' data-info.

        Raises
        ------
        AggregationError
            In case (some of) the clients' data info is invalid, or
            incompatible. Send CancelTraining to all clients before
            raising.
        """
        fields = self.model.required_data_info  # revise: add optimizer, etc.
        if not fields:
            return
        # Collect required metadata from clients.
        query = messaging.MetadataQuery(list(fields))
        await self.netwk.broadcast_message(query)
        replies = await self._collect_results(
            self.netwk.client_names,
            msgtype=messaging.MetadataReply,
            context="Metadata collection",
        )
        clients_data_info = {
            client: reply.data_info for client, reply in replies.items()
        }
        # Try aggregating the input data_info.
        try:
            info = aggregate_clients_data_info(clients_data_info, fields)
        # In case of failure, cancel training, notify clients, log and raise.
        except AggregationError as exc:
            messages = {
                client: messaging.CancelTraining(reason)
                for client, reason in exc.messages.items()
            }
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
        RuntimeError
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
        results: Dict[str, MessageT] = {}
        errors: Dict[str, str] = {}
        for client, reply in replies.items():
            if issubclass(reply.message_cls, msgtype):
                results[client] = reply.deserialize()
            elif issubclass(reply.message_cls, messaging.Error):
                err_msg = reply.deserialize().message
                errors[client] = f"{context} failed: {err_msg}"
            else:
                errors[client] = f"Unexpected message: {reply.message_cls}"
        # If any client has failed to send proper results, raise.
        # future: modularize errors-handling behaviour
        if errors:
            err_msg = f"{context} failed for another client."
            messages: Dict[str, messaging.Message] = {
                client: messaging.CancelTraining(errors.get(client, err_msg))
                for client in self.netwk.client_names
            }
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
        params: Dict[str, Any] = {
            "rounds": config.rounds,
            "batches": config.training.batch_cfg,
            "n_epoch": config.training.n_epoch,
            "n_steps": config.training.n_steps,
            **dataclasses.asdict(config.privacy),
        }
        message = messaging.PrivacyRequest(**params)
        await self.netwk.broadcast_message(message)
        self.logger.info("Waiting for clients' responses.")
        await self._collect_results(
            clients=self.netwk.client_names,
            msgtype=messaging.PrivacyReply,
            context="Privacy initialization",
        )
        self.logger.info("Privacy requests were processed by clients.")

    async def setup_secagg(
        self,
        clients: Optional[Set[str]] = None,
    ) -> None:
        """Run a setup protocol for SecAgg.

        Parameters
        ----------
        clients:
            Optional set of clients to restrict the setup to which.
        """
        self.logger.info("Setting up SecAgg afresh.")
        assert self.secagg is not None
        try:
            self._decrypter = await self.secagg.setup_decrypter(
                netwk=self.netwk, clients=clients
            )
        except RuntimeError as exc:
            error = (
                f"An exception was raised while setting up SecAgg: {repr(exc)}"
            )
            self.logger.error(error)
            await self.netwk.broadcast_message(messaging.CancelTraining(error))
            raise RuntimeError(error) from exc
        self._secagg_peers = (
            self.netwk.client_names if clients is None else clients
        )

    def _aggregate_secagg_replies(
        self,
        replies: Mapping[str, secagg_messaging.SecaggMessage[MessageT]],
    ) -> MessageT:
        """Secure-Aggregate (and decrypt) client-issued encrypted messages."""
        assert self._decrypter is not None
        return secagg_messaging.aggregate_secagg_messages(
            replies, decrypter=self._decrypter
        )

    async def fairness_round(
        self,
        round_i: int,
        fairness_cfg: FairnessConfig,
        force_run: bool = False,
    ) -> None:
        """Orchestrate a fairness round, when configured to do so.

        If fairness is not set, or if `round_i` is to be skipped based
        on `fairness_cfg.frequency`, do nothing.

        Parameters
        ----------
        round_i:
            Index of the latest training round (start at 0).
        fairness_cfg:
            FairnessConfig dataclass instance wrapping data-batching
            and computational effort constraints hyper-parameters for
            fairness evaluation.
        force_run:
            Whether to disregard `fairness_cfg.frequency` and run the
            round (provided a fairness controller is setup).
        """
        # Early exit when fairness is not set or the round is to be skipped.
        if self.fairness is None:
            return
        if (round_i % fairness_cfg.frequency) and not force_run:
            return
        # Run SecAgg setup when needed.
        self.logger.info("Initiating fairness-enforcing round %s", round_i)
        clients = self.netwk.client_names  # FUTURE: enable sampling(?)
        if self.secagg is not None and clients.difference(self._secagg_peers):
            await self.setup_secagg(clients)
        # Send a query to clients, including model weights when required.
        query = messaging.FairnessQuery(
            round_i=round_i,
            batch_size=fairness_cfg.batch_size,
            n_batch=fairness_cfg.n_batch,
            thresh=fairness_cfg.thresh,
            weights=None,
        )
        await self._send_request_with_optional_weights(query, clients)
        # Await, (secure-)aggregate and process fairness measures.
        metrics = await self.fairness.run_fairness_round(
            netwk=self.netwk,
            secagg=self._decrypter,
        )
        # Optionally save computed fairness metrics.
        if self.ckptr is not None:
            self.ckptr.save_metrics(
                metrics=metrics,
                prefix="fairness_metrics",
                append=bool(query.round_i),
                timestamp=f"round_{query.round_i}",
            )

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
        # Select participating clients. Run SecAgg setup when needed.
        self.logger.info("Initiating training round %s", round_i)
        clients = self._select_training_round_participants()
        if self.secagg is not None and clients.difference(self._secagg_peers):
            await self.setup_secagg(clients)
        # Send training instructions and await results.
        await self._send_training_instructions(clients, round_i, train_cfg)
        self.logger.info("Awaiting clients' training results.")
        if self._decrypter is None:
            results = await self._collect_results(
                clients, messaging.TrainReply, "training"
            )
        else:
            secagg_results = await self._collect_results(
                clients, secagg_messaging.SecaggTrainReply, "training"
            )
            results = {
                "aggregated": self._aggregate_secagg_replies(secagg_results)
            }
        # Aggregate client-wise results and update the global model.
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
        # Set up the base training request.
        msg_light = messaging.TrainRequest(
            round_i=round_i,
            weights=None,
            aux_var=self.optim.collect_aux_var(),
            **train_cfg.message_params,
        )
        # Send it to clients, sparingly joining model weights.
        await self._send_request_with_optional_weights(msg_light, clients)

    async def _send_request_with_optional_weights(
        self,
        msg_light: Union[
            messaging.TrainRequest,
            messaging.EvaluationRequest,
            messaging.FairnessQuery,
        ],
        clients: Set[str],
    ) -> None:
        """Send a request to clients, sparingly adding model weights to it.

        Transmit the input message to all clients, adding a copy of the
        global model weights for clients that do not already hold them.

        Parameters
        ----------
        msg_light:
            Message to send, with a 'weights' field left to None.
        clients:
            Name of the clients to whom the message is addressed.
        """
        # Identify clients that do not already hold latest model weights.
        needs_weights = clients.difference(self._clients_holding_latest_model)
        # If any client does not hold latest weights, ensure they get it.
        if needs_weights:
            msg_heavy = copy.copy(msg_light)
            msg_heavy.weights = self.model.get_weights(trainable=True)
            messages = {
                client: msg_heavy if client in needs_weights else msg_light
                for client in clients
            }
            await self.netwk.send_messages(messages)
            self._clients_holding_latest_model.update(needs_weights)
        # If no client requires weights, do not even access them.
        else:
            await self.netwk.broadcast_message(msg_light, clients)

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
        # Unpack, aggregate and finally process optimizer auxiliary variables.
        aux_var: Dict[str, AuxVar] = {}
        for msg in results.values():
            for key, aux in msg.aux_var.items():
                aux_var[key] = aux_var.get(key, 0) + aux
        self.optim.process_aux_var(aux_var)
        # Compute aggregated "gradients" (updates) and apply them to the model.
        updates = sum(msg.updates for msg in results.values())
        gradients = self.aggrg.finalize_updates(updates)
        self.optim.apply_gradients(self.model, gradients)
        # Record that no clients hold the updated model.
        self._clients_holding_latest_model.clear()

    async def evaluation_round(
        self,
        round_i: int,
        valid_cfg: EvaluateConfig,
        force_run: bool = False,
    ) -> None:
        """Orchestrate an evaluation round, when configured to do so.

        If `round_i` is to be skipped based on `fairness_cfg.frequency`,
        do nothing.

        Parameters
        ----------
        round_i: int
            Index of the latest training round.
        valid_cfg: EvaluateConfig
            EvaluateConfig dataclass instance wrapping data-batching
            and computational effort constraints hyper-parameters.
        """
        # Early exit when the evaluation round is to be skipped.
        if (round_i % valid_cfg.frequency) and not force_run:
            return
        # Select participating clients. Run SecAgg setup when needed.
        self.logger.info("Initiating evaluation round %s", round_i)
        clients = self._select_evaluation_round_participants()
        if self.secagg is not None and clients.difference(self._secagg_peers):
            await self.setup_secagg(clients)
        # Send evaluation requests and collect clients' replies.
        await self._send_evaluation_instructions(clients, round_i, valid_cfg)
        self.logger.info("Awaiting clients' evaluation results.")
        if self._decrypter is None:
            results = await self._collect_results(
                clients, messaging.EvaluationReply, "evaluation"
            )
        else:
            secagg_results = await self._collect_results(
                clients, secagg_messaging.SecaggEvaluationReply, "evaluation"
            )
            results = {
                "aggregated": self._aggregate_secagg_replies(secagg_results)
            }
        # Compute and report aggregated evaluation metrics.
        self.logger.info("Aggregating evaluation results.")
        loss, metrics = self._aggregate_evaluation_results(results)
        self.logger.info("Averaged loss is: %s", loss)
        if metrics:
            self.logger.info(
                "Other averaged scalar metrics are: %s",
                {
                    k: f"{v:.4f}"
                    for k, v in metrics.items()
                    if isinstance(v, float)
                },
            )
        # Optionally checkpoint the model, optimizer and metrics.
        if self.ckptr:
            self._checkpoint_after_evaluation(
                metrics, results if len(results) > 1 else {}
            )
        # Record the global loss, and update the kept "best" weights.
        self._losses.append(loss)
        if loss == min(self._losses):
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
        # Set up the base evaluation request.
        msg_light = messaging.EvaluationRequest(
            round_i=round_i,
            weights=None,
            **valid_cfg.message_params,
        )
        # Send it to clients, sparingly joining model weights.
        await self._send_request_with_optional_weights(msg_light, clients)

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
        agg_states = self.metrics.get_states()
        # Iteratively update the MetricSet and loss floats based on results.
        for client, reply in results.items():
            # Case when the client reported some metrics.
            if reply.metrics:
                states = reply.metrics.copy()
                # Deal with loss metric's aggregation.
                s_loss = states.pop("loss")
                assert isinstance(s_loss, MeanState)
                loss += s_loss.num_sum
                dvsr += s_loss.divisor
                # Aggregate other metrics.
                for key, val in states.items():
                    agg_states[key] += val
            # Case when the client only reported the aggregated local loss.
            else:
                self.logger.info(
                    "Client %s refused to share their local metrics.", client
                )
                loss += reply.loss
                dvsr += reply.n_steps
        # Compute the final results.
        self.metrics.set_states(agg_states)
        metrics = self.metrics.get_result()
        loss = loss / dvsr
        metrics.setdefault("loss", loss)
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
        for client, reply in results.items():
            metrics = {"loss": reply.loss}
            if reply.metrics:
                self.metrics.set_states(reply.metrics)
                metrics.update(self.metrics.get_result())
            self.ckptr.save_metrics(
                metrics=metrics,
                prefix=f"metrics_{client}",
                append=bool(self._losses),
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
            early_stop.update(self._losses[-1])
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
            loss=min(self._losses, default=float("nan")),
            rounds=rounds,
        )
        self.logger.info("Notifying clients that training is over.")
        await self.netwk.broadcast_message(message)
        if self.ckptr:
            path = f"{self.ckptr.folder}/model_state_best.json"
            self.logger.info("Checkpointing final weights under %s.", path)
            self.model.set_weights(message.weights)
            self.ckptr.save_model(self.model, timestamp="best")
