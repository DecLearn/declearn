# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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
import os
import warnings
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
from declearn.client_sampler import (
    ClientSampler,
    ClientSamplerConfig,
    DefaultClientSampler,
)
from declearn.communication import NetworkServerConfig
from declearn.communication.api import NetworkServer
from declearn.fairness.api import FairnessControllerServer
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
    IncompatibleConfigsError,
    ServerUtils,
    aggregate_clients_data_info,
    dispatch_replies,
)
from declearn.messaging import (
    MetadataReply,
    MultiModelFairnessQuery,
    SerializedMessage,
)
from declearn.metrics import MetricInputType, MetricSet
from declearn.metrics._mean import MeanState
from declearn.model.api import Model, Vector
from declearn.optimizer.modules import AuxVar
from declearn.secagg import messaging as secagg_messaging
from declearn.secagg.api import Decrypter, SecaggConfigServer

__all__ = [
    "FederatedServer",
]
MessageT = TypeVar("MessageT", bound=messaging.Message)


class FederatedServer:
    """Server-side Federated Learning orchestrating class.

    Notes
    -----
    You can access and configure this class logger using
    `logger = logging.getLogger("declearn.server")`, and then adjust it as
    needed (e.g. `logger.setLevel(...)`).
    """

    # one-too-many attribute; pylint: disable=too-many-instance-attributes
    # pylint: disable-next=too-many-positional-arguments
    # TODO for 2.10 : remove deprecated "logger" argument
    def __init__(  # noqa: PLR0913
        self,
        model: Union[
            List[Union[Model, str, Dict[str, Any]]], Model, str, Dict[str, Any]
        ],
        netwk: Union[NetworkServer, NetworkServerConfig, Dict[str, Any], str],
        optim: Union[FLOptimConfig, str, Dict[str, Any]],
        metrics: Union[MetricSet, List[MetricInputType], None] = None,
        client_sampler: Union[
            ClientSampler, ClientSamplerConfig, Dict[str, Any], None
        ] = None,
        secagg: Union[SecaggConfigServer, Dict[str, Any], None] = None,
        checkpoint: Union[Checkpointer, Dict[str, Any], str, None] = None,
        logger: Union[logging.Logger, str, None] = None,
    ) -> None:
        """Instantiate the orchestrating server for a federated learning task.

        Parameters
        ----------
        model: An instance of Model, dict or str, or a list of  Model, dict or
            str.
            Model instance, that may be serialized as an ObjectConfig,
            a config dict or a JSON file the path to which is provided.
        netwk: NetworkServer or NetworkServerConfig or dict or str
            NetworkServer communication endpoint instance, or configuration
            dict, dataclass or path to a TOML file enabling its instantiation.
            In the latter three cases, the object's default logger will
            be set to that of this `FederatedServer`.
        optim: A list of FLOptimConfig or dict or str
            FLOptimConfig instance or instantiation dict (using
            the `from_params` method) or TOML configuration file path.
            This object specifies the optimizers to use by the clients
            and the server, as well as the client-updates aggregator.
        metrics: A list of MetricSet or list[MetricInputType] or None,
        default=None
            MetricSet instance or list of Metric instances and/or specs
            to wrap into one, defining evaluation metrics to compute in
            addition to the model's loss.
            If None, only compute and report the model's loss.
        client_sampler: ClientSampler or ClientSamplerConfig or specification
            dict or None (default). Specifies the client sampler to use in the
            federated process to select clients involved at each round.
        secagg: SecaggConfigServer or dict or None, default=None
            Optional SecAgg config and setup controller
            or dict of kwargs to set one up.
        checkpoint: Checkpointer or dict or str or None, default=None
            Optional Checkpointer instance or instantiation dict to be
            used so as to save round-wise model, optimizer and metrics.
            If a single string is provided, treat it as the checkpoint
            folder path and use default values for other parameters.
        logger: logging.Logger or str or None, default=None,
            Deprecated in v2.8, removed in v2.10.
            Not used anymore.
        """
        # Init class logger.
        if logger is not None:
            warnings.warn(
                "Argument 'logger' is deprecated and useless now, it will be "
                "removed in 2.10. "
                "To customize the instance logger, you may use instead "
                "logging utils from `declearn.utils` or the 'logging' Python "
                "module.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.logger = logging.getLogger("declearn.server")
        if not isinstance(model, List):
            model = [model]
        self.model: List[Model] = list(map(ServerUtils.parse_model, model))
        self._n_models: int = len(self.model)
        parsed_optim: FLOptimConfig = ServerUtils.parse_optim(optim)
        self.c_opt, self.optim, self.aggrg = ServerUtils.copy_optimizer_specs(
            self._n_models, parsed_optim
        )
        self.fairness: Optional[FairnessControllerServer] = (
            parsed_optim.fairness
        )
        # Assign the wrapped MetricSet.
        self.metrics = MetricSet.from_specs(metrics)
        self.netwk = ServerUtils.parse_netwk(netwk)
        # Assign a client sampler - by default one is used
        self.client_sampler = ServerUtils.parse_clisamp(
            client_sampler, logger=self.logger
        )
        # Assign an optional checkpointer.
        if checkpoint is not None:
            checkpoint = Checkpointer.from_specs(checkpoint)
        self.ckptr = checkpoint

        # Assign the optional SecAgg config and declare a Decrypter slot.
        self.secagg = ServerUtils.parse_secagg(secagg)
        self._check_clisamp_secagg_compat()
        self._decrypter: Optional[Decrypter] = None
        self._secagg_peers: Set[str] = set()
        # Set up a private attribute to record the loss values.
        self._losses: Dict[str, List[float]] = {
            f"model_{i}": [] for i in range(self._n_models)
        }
        # Set up a private attribute to record the best trainable weights
        self._best: Dict[str, Optional[Vector]] = {
            f"model_{i}": None for i in range(self._n_models)
        }
        # Set up a private attribute to prevent redundant weights sharing
        self._clients_holding_latest_model: Dict[str, Set[str]] = {
            f"model_{i}": set() for i in range(self._n_models)
        }

    def _check_clisamp_secagg_compat(self) -> None:
        """Check if instantiated client sampler config and secure aggregation
        config are compatible, if not: raises an exception

        Raises
        ------
        IncompatibleConfigsError
            In case the server client sampler and secure aggregation configs
            are incompatible
        """
        if (
            self.secagg is not None
            and not self.client_sampler.secagg_compatible
        ):
            raise IncompatibleConfigsError(
                "Secure aggregation is enabled, but the selected client "
                "sampler is not compatible with secure aggregation."
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
        config = ServerUtils.parse_configs(config)
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
        ServerUtils.check_modules_compatibility(
            self.client_sampler, config.privacy, self.fairness, self._n_models
        )
        assert isinstance(config.training, List)
        assert isinstance(config.evaluate, List)
        frequency: int = config.evaluate[0].frequency
        # Instantiate the early-stopping criterion, if any.
        early_stop: Optional[List[EarlyStopping]] = None
        if config.early_stop is not None:
            early_stop = [
                config.early_stop.instantiate() for _ in range(self._n_models)
            ]
        # Start the communications server and run the FL process.
        async with self.netwk:
            # Conduct the initialization phase.
            await self.initialization(config)
            self.client_sampler.init_clients(self.netwk.client_names)
            if self.ckptr:
                for idx in range(self._n_models):
                    self.ckptr.checkpoint(
                        self.model[idx],
                        self.optim[idx],
                        first_call=True,
                        idx=None if self._n_models == 1 else idx,
                    )
            # Iteratively run training and evaluation rounds
            round_i = 0
            while True:
                clients_train = self._select_training_round_participants()
                clients_eval = self._select_evaluation_round_participants()
                # Fairness round is made on all clients.
                await self.fairness_round(
                    round_i, config.fairness, self.netwk.client_names
                )
                round_i += 1
                await self.training_round(
                    round_i, config.training, clients_train
                )
                await self.evaluation_round(
                    round_i, config.evaluate, clients_eval
                )
                # Decide whether to keep training for at least one round.
                if not self._keep_training(round_i, config.rounds, early_stop):
                    break
            # When checkpointing, force evaluating the last model on everyone
            if self.ckptr is not None:
                if round_i % frequency:
                    await self.evaluation_round(
                        round_i,
                        config.evaluate,
                        self.netwk.client_names,
                        force_run=True,
                    )
                await self.fairness_round(
                    round_i,
                    config.fairness,
                    self.netwk.client_names,
                    force_run=True,
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
        Send instructions to clients to set up their models and optimizers.
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
            regst_cfg.min_clients,
            regst_cfg.max_clients,
            regst_cfg.timeout,
        )
        self.logger.info("Clients' registration is now complete.")
        # When needed, prompt clients for metadata and process them.
        await self._require_and_process_data_info()
        # Serialize initialization information and send it to clients.
        message = messaging.InitRequest(
            model=self.model,
            optim=self.c_opt,
            aggrg=self.aggrg,
            metrics=self.metrics.get_config()["metrics"],
            dpsgd=config.privacy is not None,
            secagg=None if self.secagg is None else self.secagg.secagg_type,
            fairness=self.fairness is not None,
        )
        await self.netwk.broadcast_message(message)
        self.logger.info("Initialization instructions sent successfully.")
        # Await a confirmation from clients that initialization went well.
        self.logger.info("Waiting for clients' responses.")
        await self._collect_results(
            clients=self.netwk.client_names,
            msgtype=messaging.InitReply,
            context="Initialization",
        )
        # If local differential privacy is configured, set it up.
        if config.privacy is not None:
            await self._initialize_dpsgd(config)
        if self.fairness is not None:
            # When SecAgg is to be used, setup controllers first.
            if self.secagg is not None:
                await self.setup_secagg()
            # Call the setup routine of the held fairness controller.
            self.aggrg[0] = await self.fairness.setup_fairness(
                netwk=self.netwk,
                aggregator=self.aggrg[0],
                secagg=self._decrypter,
            )
        self.logger.info("Initialization was successful.")

    async def _require_and_process_data_info(
        self,
    ) -> None:
        """Collect, validate, aggregate and make use of clients' data-info

        Raises
        ------
        AggregationError
            In case (some of) the clients' data info is invalid, or
            incompatible. Send CancelTraining to all clients before
            raising.
        """
        fields = {
            f"model_{idx}": model.required_data_info
            for idx, model in enumerate(self.model)
        }
        # revise: add optimizer, etc.
        if not any(fields.values()):
            return
        # Collect required metadata from clients.
        query = messaging.MetadataQuery(
            fields=fields  # type: ignore [arg-type]
        )
        await self.netwk.broadcast_message(query)
        replies = await self._collect_results(
            self.netwk.client_names,
            msgtype=MetadataReply,
            context="Metadata collection",
        )
        # Process each model independently
        for idx in range(self._n_models):
            key = f"model_{idx}"
            client_info: Dict[str, Dict] = {
                client: reply.data_info[key]
                for client, reply in replies.items()
            }
            try:
                aggregated_info: Dict[str, Any] = aggregate_clients_data_info(
                    client_info,
                    fields[key],
                )
            except AggregationError as exc:
                messages = {
                    client: messaging.CancelTraining(reason)
                    for client, reason in exc.messages.items()
                }
                await self.netwk.send_messages(messages)
                self.logger.error(exc.error)
                raise exc
            self.model[idx].initialize(aggregated_info)

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
        replies: Dict[str, SerializedMessage]
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
        # future: modularize errors-handling behavior
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
        """Send a differential privacy setup request to all
        registered clients."""

        self.logger.info("Sending privacy requests to all clients.")
        assert config.privacy is not None  # else this method is not called
        assert isinstance(config.training, List)  # for mypy
        training_cfg = config.training[0]
        params: Dict[str, Any] = {
            "rounds": config.rounds,
            "batches": training_cfg.batch_cfg,
            "n_epoch": training_cfg.n_epoch,
            "n_steps": training_cfg.n_steps,
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
        replies: Union[
            Mapping[str, secagg_messaging.SecaggMessage[MessageT]],
        ],
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
        clients: Set[str],
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
        clients:
            Set of clients taking part to this fairness round.
        force_run:
            Whether to disregard `fairness_cfg.frequency` and run the
            round (provided a fairness controller is setup).
        """
        if self.fairness is None:
            return
        if (round_i % fairness_cfg.frequency) and not force_run:
            return
        # Run SecAgg setup when needed.
        self.logger.info("Initiating fairness-enforcing round %s", round_i)
        if self.secagg is not None and clients.difference(self._secagg_peers):
            await self.setup_secagg(clients)
        # Send a query to clients, including model weights when required.
        query = messaging.MultiModelFairnessQuery(
            requests=[
                messaging.FairnessQuery(
                    round_i=round_i,
                    batch_size=fairness_cfg.batch_size,
                    n_batch=fairness_cfg.n_batch,
                    thresh=fairness_cfg.thresh,
                    weights=None,
                )
                for _ in self.model
            ]
        )
        await self._send_request_with_optional_weights(query, clients)
        # Optionally save computed fairness metrics
        metrics = await self.fairness.run_fairness_round(
            netwk=self.netwk,
            secagg=self._decrypter,
        )
        if self.ckptr is not None:
            self.ckptr.save_metrics(
                metrics=metrics,
                prefix="fairness_metrics",
                append=bool(round_i),
                timestamp=f"round_{round_i}",
            )

    async def training_round(  # noqa: PLR0913
        self,
        round_i: int,
        train_cfgs: List[TrainingConfig],
        clients: Set[str],
    ) -> None:
        """Orchestrate a training round.

        Parameters
        ----------
        round_i: int
            Index of the training round.
        train_cfg: List[TrainingConfig]
            List of TrainingConfig dataclass instances wrapping data-batching
            and computational effort constraints hyper-parameters for every
            model.
        clients:
            Set of clients taking part to this training round.
        """
        # Select participating clients. Run SecAgg setup when needed.
        self.logger.info(
            "Initiating training round %s",
            round_i,
        )
        if self.secagg is not None and clients.difference(self._secagg_peers):
            await self.setup_secagg(clients)
        # Send training instructions and await results.
        await self._send_training_instructions(clients, round_i, train_cfgs)
        self.logger.info("Awaiting clients' training results.")
        results = (
            await self._process_model_updates(clients)
            if self._decrypter is None
            else await self._process_secagg_model_updates(clients)
        )
        # Aggregate client-wise results and update the global model.
        self.logger.info("Conducting server-side optimization.")
        for idx, result in enumerate(results):
            self._conduct_global_update(
                result,
                idx=idx,
            )

    async def _process_model_updates(self, clients):
        """Process non-SecAgg multi-model training replies."""

        multi_model_results = await self._collect_results(
            clients, messaging.MultiModelTrainReply, "training"
        )
        results_per_model = dispatch_replies(
            multi_model_results, lambda reply: reply.replies
        )
        # NOTE update the client sampler in the case of training
        # one single model
        if self._n_models == 1:
            result = results_per_model[0]
            self.client_sampler.update(result, self.model[0])

        return results_per_model

    async def _process_secagg_model_updates(self, clients):
        """Process the model updates step in the case where the replies of
        all clients are secagged"""

        secagg_results = await self._collect_results(
            clients, secagg_messaging.MultiModelSecaggTrainReply, "training"
        )
        results_per_model = dispatch_replies(
            secagg_results,
            lambda reply: reply.replies,
        )
        processed_results = []
        for idx, secagg_result in enumerate(results_per_model):
            aggregated_result = self._aggregate_secagg_replies(
                secagg_result,
            )

            # In the SecAgg case, provide the sampler with each # participating
            # client associated with the aggregated result.
            client_results = {client: aggregated_result for client in clients}
            if self._n_models == 1:
                self.client_sampler.update(
                    client_results,
                    self.model[idx],
                )
            processed_results.append({"aggregated": aggregated_result})
        return processed_results

    def _select_training_round_participants(
        self,
    ) -> Set[str]:
        """Return the names of clients that should participate in the round."""
        sampled_clients = self.client_sampler.run()
        if not isinstance(self.client_sampler, DefaultClientSampler):
            self.logger.debug(
                f"Sampled clients for train round are : {sampled_clients}"
            )
        return sampled_clients

    async def _send_training_instructions(
        self, clients: Set[str], round_i: int, train_cfg: List[TrainingConfig]
    ) -> None:
        """Send training instructions to selected clients.

        Parameters
        ----------
        clients: set[str]
            Names of the clients participating in the training round.
        round_i: int
            Index of the training round.
        train_cfg: List[TrainingConfig]
            A list of TrainingConfig dataclass instance wrapping data-batching
            and computational effort constraints hyper-parameters.
        """
        # Set up the base training request.
        requests = [
            messaging.TrainRequest(
                round_i=round_i,
                weights=None,
                aux_var=self.optim[idx].collect_aux_var(),
                **train_cfg[idx].message_params,
            )
            for idx in range(self._n_models)
        ]
        msg_light = messaging.MultiModelTrainRequest(requests)
        # Send it to clients, sparingly joining model weights.
        await self._send_request_with_optional_weights(
            msg_light,
            clients,
        )

    async def _send_request_with_optional_weights(
        self,
        msg_light: Union[
            messaging.MultiModelTrainRequest,
            messaging.MultiModelEvaluationRequest,
            MultiModelFairnessQuery,
        ],
        clients: Set[str],
    ) -> None:
        """Send a multi-model request, adding missing model weights
        when needed."""

        # Determine which clients need the weights for each model.
        needs_weights = {}
        for idx in range(len(msg_light.requests)):
            holders = self._clients_holding_latest_model[f"model_{idx}"]
            needs_weights[idx] = clients.difference(holders)
        # Fetch each model's weights at most once.
        model_weights = {}
        for idx, clients_needing_weights in needs_weights.items():
            if clients_needing_weights:
                model_weights[idx] = self.model[idx].get_weights(
                    trainable=True,
                )
        messages = await ServerUtils.build_client_messages(
            msg_light, clients, needs_weights, model_weights
        )
        await self.netwk.send_messages(messages)
        # Only mark weights as held after a successful send.
        for idx, clients_needing_weights in needs_weights.items():
            self._clients_holding_latest_model[f"model_{idx}"].update(
                clients_needing_weights
            )

    def _conduct_global_update(
        self,
        results: Dict[str, messaging.TrainReply],
        idx: int,
    ) -> None:
        """Use training results from clients to update the global model.

        Parameters
        ----------
        results: dict[str, TrainReply]
            Client-wise TrainReply messages sent after a training round.
        idx: int
            Index of the model on which the updates must be conducted.
        """
        # Unpack, aggregate and finally process optimizer auxiliary
        # variables.
        aux_var: Dict[str, AuxVar] = {}
        for msg in results.values():
            for key, aux in msg.aux_var.items():
                aux_var[key] = aux_var.get(key, 0) + aux

        self.optim[idx].process_aux_var(aux_var)
        # Compute aggregated "gradients" (updates) and apply them to
        # the model.
        updates = sum(msg.updates for msg in results.values())
        gradients = self.aggrg[idx].finalize_updates(updates)
        self.optim[idx].apply_gradients(self.model[idx], gradients)
        # Record that no clients hold the updated model.
        self._clients_holding_latest_model[f"model_{idx}"].clear()

    async def evaluation_round(
        self,
        round_i: int,
        valid_cfg: List[EvaluateConfig],
        clients: Set[str],
        force_run: bool = False,
    ) -> None:
        """Orchestrate an evaluation round, when configured to do so.

        If `round_i` is to be skipped based on `fairness_cfg.frequency`,
        do nothing.

        Parameters
        ----------
        round_i: int
            Index of the latest training round.
        valid_cfg: List[EvaluateConfig]
            List of EvaluateConfig dataclass instances wrapping data-batching
            and computational effort constraints hyper-parameters for
            every trainable model.
        clients: Set[str]
            Set of clients used for evaluation during the round.
        force_run: bool, default=False
            Flag to wether force run the evaluation round.
        """
        if (round_i % valid_cfg[0].frequency) and not force_run:
            return
        # Select participating clients. Run SecAgg setup when needed.
        self.logger.info("Initiating evaluation round %s", round_i)
        if self.secagg is not None and clients.difference(self._secagg_peers):
            await self.setup_secagg(clients)
        # Send evaluation requests and collect clients' replies.
        await self._send_evaluation_instructions(clients, round_i, valid_cfg)
        self.logger.info("Awaiting clients' evaluation results.")
        results = []
        if self._decrypter is None:
            response = await self._collect_results(
                clients, messaging.MultiModelEvaluationReply, "evaluation"
            )
            results = dispatch_replies(response, lambda reply: reply.replies)
        else:
            multi_model_results = await self._collect_results(
                clients,
                secagg_messaging.MultiModelSecaggEvaluationReply,
                "evaluation",
            )
            results_per_model: list[
                dict[str, secagg_messaging.SecaggEvaluationReply]
            ] = dispatch_replies(
                multi_model_results,
                lambda reply: reply.replies,
            )
            for model_results in results_per_model:
                aggregated_result: messaging.EvaluationReply = (
                    self._aggregate_secagg_replies(model_results)
                )
                results.append({"aggregated": aggregated_result})
        # Compute and report aggregated evaluation metrics.
        self.logger.info("Aggregating evaluation results.")
        for idx, result in enumerate(results):
            loss, metrics = self._aggregate_evaluation_results(result)
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
                    metrics, result if len(result) > 1 else {}, idx
                )
            # Record the global loss, and update the kept
            # "best" trainable weights (per model)
            self._losses[f"model_{idx}"].append(loss)
            if loss == min(self._losses[f"model_{idx}"]):
                self._best[f"model_{idx}"] = self.model[idx].get_weights(
                    trainable=True
                )

    def _select_evaluation_round_participants(
        self,
    ) -> Set[str]:
        """Return the names of clients that should participate in the round."""
        # FUTURE: implement client sampling for evaluation rounds ?
        return self.netwk.client_names

    async def _send_evaluation_instructions(
        self,
        clients: Set[str],
        round_i: int,
        valid_cfg: List[EvaluateConfig],
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
        msg_light = messaging.MultiModelEvaluationRequest(
            requests=[
                messaging.EvaluationRequest(
                    round_i=round_i,
                    weights=None,
                    **evaluation_request.message_params,
                )
                for evaluation_request in valid_cfg
            ]
        )
        await self._send_request_with_optional_weights(
            msg_light,
            clients,
        )

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
        idx: int, default=0.
            Zero-based index of the model to which the aggregation
            to be performed.

        Returns
        -------
        loss: float
            The aggregated loss score computed from clients' ones.
        metrics: dict[str, (float | np.ndarray)]
            The aggregated evaluation metrics computes from clients' ones.
        """
        loss = 0.0
        dvsr = 0.0
        # Reset the local MetricSet and set up ad hoc variables for the loss.
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
                    "Client %s refused to share their local metrics.",
                    client,
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
        idx: int = 0,
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
        idx : int, default=0.
            Zero-based index of the model to which the aggregation
            to be performed.
        """
        # This method only works when a checkpointer is used.
        if self.ckptr is None:
            raise RuntimeError(
                "`_checkpoint_after_evaluation` was called without "
                "the FederatedServer having a Checkpointer."
            )
        # Checkpoint the model, optimizer and global evaluation metrics.
        timestamp = self.ckptr.checkpoint(
            model=self.model[idx],
            optimizer=self.optim[idx],
            metrics=metrics,
            idx=None if self._n_models == 1 else idx,
        )
        # Checkpoint the client-wise metrics (or at least their loss).
        # Use the same timestamp label as for global metrics and states.
        for client, reply in results.items():
            metrics = {"loss": reply.loss}
            if reply.metrics:
                if self.metrics is not None:
                    self.metrics.set_states(reply.metrics)
                    metrics.update(self.metrics.get_result())
            self.ckptr.save_metrics(
                metrics=metrics,
                prefix=f"metrics_{client}",
                append=bool(self._losses[f"model_{idx}"]),
                timestamp=timestamp,
                idx=None if self._n_models == 1 else idx,
            )

    def _keep_training(
        self,
        round_i: int,
        rounds: int,
        early_stop: Optional[List[EarlyStopping]],
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
            for idx, loss_list in enumerate(self._losses.values()):
                early_stop[idx].update(loss_list[-1])

            if any(
                [
                    early_stop[idx].keep_training
                    for idx in range(self._n_models)
                ]
            ):
                return True

            self.logger.info("Early stopping criterion reached.")
            return False
        return True

    async def stop_training(self, rounds: int) -> None:
        """Notify clients that training is over and send final information

        Parameters
        ----------
        rounds: int
            Number of training rounds taken until now.
        """
        self.logger.info("Recovering weights that yielded the lowest loss")
        weights = {}
        for idx, model in enumerate(self.model):
            best_weights = self._best.get(f"model_{idx}")
            weights[f"model_{idx}"] = (
                best_weights
                if best_weights is not None
                else model.get_weights(trainable=True)
            )
        losses = {
            f"model_{idx}": min(
                self._losses[f"model_{idx}"],
                default=float("nan"),
            )
            for idx in range(self._n_models)
        }
        message = messaging.StopTraining(
            weights=weights,
            loss=losses,
            rounds=rounds,
        )
        self.logger.info("Notifying clients that training is over.")
        await self.netwk.broadcast_message(message)
        if self.ckptr:
            for idx, model in enumerate(self.model):
                model.set_weights(
                    weights[f"model_{idx}"],
                    trainable=True,
                )
                self.ckptr.save_model(
                    model,
                    timestamp="best",
                    idx=None if self._n_models == 1 else idx,
                )
                path = os.path.join(self.ckptr.folder, "model_state_best.mpk")
                self.logger.info(
                    "Checkpointed for model %d.under %s", idx + 1, path
                )
