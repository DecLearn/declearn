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
"""Server utilities class for FederatedServer."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Union

import declearn
from declearn import messaging
from declearn.client_sampler import (
    ClientSampler,
    ClientSamplerConfig,
    DefaultClientSampler,
    instantiate_client_sampler,
)
from declearn.communication import NetworkServerConfig
from declearn.communication.api import NetworkServer
from declearn.main.utils import (
    copy_requests_with_weights,
)
from declearn.model.api import Model
from declearn.optimizer import Optimizer
from declearn.secagg import parse_secagg_config_server
from declearn.secagg.api import SecaggConfigServer


class ServerUtils:
    """Static helpers to parse inputs used to set up a federated server."""

    @staticmethod
    def parse_configs(
        config: Union[
            declearn.main.config.FLRunConfig, str, Dict[str, Any], List[Any]
        ],
    ) -> declearn.main.config.FLRunConfig:
        """Parse a federated-learning configuration.

        Parameters
        ----------
        config:
            Either an `FLRunConfig`, a dict of keyword arguments, or the
            path to a TOML configuration file.

        Returns
        -------
        config:
            The parsed `FLRunConfig`.
        """
        if isinstance(config, declearn.main.config.FLRunConfig):
            return config
        if isinstance(config, dict):
            return declearn.main.config.FLRunConfig.from_params(**config)
        if isinstance(config, str):
            return declearn.main.config.FLRunConfig.from_toml(config)

        raise TypeError(
            "'config' should be an FLRunConfig instance, a dict of "
            "keyword arguments, or the path to a TOML file; got "
            f"'{type(config).__name__}'."
        )

    @staticmethod
    def parse_model(
        model: Union[Model, str, Dict[str, Any]],
    ) -> Model:
        """Parse the `model` instantiation argument.

        Parameters
        ----------
        model:
            Either a `Model` instance, a dict-shaped model configuration,
            or the path to a JSON file containing such a configuration.

        Returns
        -------
        model:
            The instantiated `Model`.
        """
        if isinstance(model, Model):
            return model

        if isinstance(model, str):
            try:
                with open(model, "r", encoding="utf-8") as file:
                    model = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                raise TypeError(
                    "Failed to load or JSON-deserialize the model "
                    f"configuration file '{model}'."
                ) from exc

        if isinstance(model, dict):
            try:
                return Model.from_config(model, allow_bin=False)
            except Exception as exc:
                raise TypeError(
                    "Failed to instantiate a Model from the provided "
                    "'model' configuration dictionary."
                ) from exc

        raise TypeError(
            "'model' should be a declearn Model instance, a model "
            "configuration dict, or the path to a JSON configuration "
            f"file; got '{type(model).__name__}'."
        )

    @staticmethod
    def parse_optim(
        optim: Union[declearn.main.config.FLOptimConfig, str, Dict[str, Any]],
    ) -> declearn.main.config.FLOptimConfig:
        """Parse the `optim` instantiation argument.

        Parameters
        ----------
        optim:
            Either an `FLOptimConfig`, a dict of keyword arguments, or
            the path to a TOML configuration file.

        Returns
        -------
        optim:
            The parsed `FLOptimConfig`.
        """
        if isinstance(optim, declearn.main.config.FLOptimConfig):
            return optim
        if isinstance(optim, str):
            return declearn.main.config.FLOptimConfig.from_toml(optim)
        if isinstance(optim, dict):
            return declearn.main.config.FLOptimConfig.from_params(**optim)

        raise TypeError(
            "'optim' should be an FLOptimConfig instance, a dict of "
            "keyword arguments, or the path to a TOML file; got "
            f"'{type(optim).__name__}'."
        )

    @staticmethod
    def copy_optimizer_specs(
        n_models: int, parsed_optim: declearn.main.config.FLOptimConfig
    ) -> tuple[
        List[Optimizer], List[Optimizer], List[declearn.aggregator.Aggregator]
    ]:
        """Makes fresh copies of the necessary optimizer specs, specifically
        client optimizer, server optimizer and the aggregator.

        Parameters
        ----------
        n_models: int
            Number of models to be trained.

        parsed_optim: FLOptimConfig
            Parsed instance of the optimizer configuration.

        Returns
        -------
        tuple: tuple[List[Optimizer], List[Optimizer], List]
            Copies of the optimizer specs (ordered as (c_opt, s_opt, aggrg))
        """
        c_opt, s_opt, aggrg = (
            parsed_optim.client_opt,
            parsed_optim.server_opt,
            parsed_optim.aggregator,
        )
        # copy the aggregator anyways
        aggregators: List[declearn.aggregator.Aggregator] = [
            aggrg.from_config(aggrg.get_config()) for _ in range(n_models)
        ]
        # user already provided a list of client & server optimizers
        if isinstance(c_opt, List) and isinstance(s_opt, List):
            if len(c_opt) > 1 and len(s_opt) > 1:
                return c_opt, s_opt, aggregators
        client_opt: Optimizer = (
            c_opt if isinstance(c_opt, Optimizer) else c_opt[0]
        )
        server_opt: Optimizer = (
            s_opt if isinstance(s_opt, Optimizer) else s_opt[0]
        )
        client_optimizers: List[Optimizer] = (
            [
                client_opt.from_config(client_opt.get_config())
                for _ in range(n_models)
            ]
            if not isinstance(client_opt, List)
            else client_opt
        )
        server_optimizers: List[Optimizer] = (
            [
                server_opt.from_config(server_opt.get_config())
                for _ in range(n_models)
            ]
            if not isinstance(parsed_optim.server_opt, List)
            else parsed_optim.server_opt
        )
        aggregators = [
            aggrg.from_config(aggrg.get_config()) for _ in range(n_models)
        ]
        return client_optimizers, server_optimizers, aggregators

    @staticmethod
    def parse_netwk(
        netwk: Union[NetworkServer, NetworkServerConfig, Dict[str, Any], str],
    ) -> NetworkServer:
        """Parse the `netwk` instantiation argument.

        Parameters
        ----------
        netwk:
            Either a `NetworkServer`, a `NetworkServerConfig`, a dict of
            configuration keyword arguments, or the path to a TOML file.

        Returns
        -------
        netwk:
            The instantiated `NetworkServer`.
        """
        if isinstance(netwk, NetworkServer):
            return netwk

        if isinstance(netwk, NetworkServerConfig):
            config = netwk
        elif isinstance(netwk, str):
            config = NetworkServerConfig.from_toml(netwk)
        elif isinstance(netwk, dict):
            config = NetworkServerConfig.from_params(**netwk)
        else:
            raise TypeError(
                "'netwk' should be a NetworkServer instance, a "
                "NetworkServerConfig, a configuration dict, or the path "
                f"to a TOML file; got '{type(netwk).__name__}'."
            )

        return config.build_server()

    @staticmethod
    def parse_clisamp(
        client_sampler: Union[
            ClientSampler, ClientSamplerConfig, Dict[str, Any], None
        ],
        logger: logging.Logger,
    ) -> ClientSampler:
        """Parse the `client_sampler` instantiation argument.

        If `None`, return a `DefaultClientSampler`, which selects every
        registered client at each round.

        Parameters
        ----------
        client_sampler:
            Either a `ClientSampler`, a `ClientSamplerConfig`, a dict of
            sampler keyword arguments, or `None`.
        logger:
            Logger used to report the selected sampling strategy.

        Returns
        -------
        client_sampler:
            The instantiated `ClientSampler`.
        """
        if client_sampler is None:
            parsed_sampler: ClientSampler = DefaultClientSampler()
        elif isinstance(client_sampler, ClientSampler):
            parsed_sampler = client_sampler
        elif isinstance(client_sampler, ClientSamplerConfig):
            parsed_sampler = client_sampler.build()
        elif isinstance(client_sampler, dict):
            parsed_sampler = instantiate_client_sampler(**client_sampler)
        else:
            raise TypeError(
                "'client_sampler' should be a ClientSampler instance, a "
                "ClientSamplerConfig, a sampler configuration dict, or "
                f"None; got '{type(client_sampler).__name__}'."
            )

        if isinstance(parsed_sampler, DefaultClientSampler):
            logger.info(
                "Default client sampler selected: all clients will be "
                "involved in each round."
            )
        else:
            logger.info(
                "Selected client sampler: "
                f"'{parsed_sampler.__class__.__name__}'."
            )

        return parsed_sampler

    @staticmethod
    def parse_secagg(
        secagg: Union[SecaggConfigServer, Dict[str, Any], None],
    ) -> Optional[SecaggConfigServer]:
        """Parse the `secagg` instantiation argument.

        Parameters
        ----------
        secagg:
            Either a `SecaggConfigServer`, a dict of keyword arguments,
            or `None` to disable secure aggregation.

        Returns
        -------
        secagg:
            The parsed `SecaggConfigServer`, or `None`.
        """
        if secagg is None:
            return None
        if isinstance(secagg, SecaggConfigServer):
            return secagg
        if isinstance(secagg, dict):
            try:
                return parse_secagg_config_server(**secagg)
            except Exception as exc:
                raise TypeError(
                    "Failed to parse the provided 'secagg' configuration."
                ) from exc

        raise TypeError(
            "'secagg' should be a SecaggConfigServer instance, a dict of "
            "keyword arguments, or None; got "
            f"'{type(secagg).__name__}'."
        )

    @staticmethod
    async def build_client_messages(
        msg_light: Union[
            messaging.MultiModelTrainRequest,
            messaging.MultiModelEvaluationRequest,
            messaging.MultiModelFairnessQuery,
        ],
        clients: Set[str],
        needs_weights: Dict,
        model_weights: Dict,
    ) -> Dict:
        """Method that builds the client messages to send relevant requests
        with necessary attached optional weights, this method is called in the
        server `server._send_request_with_optional_weights` and serves useful
        to enforce request message type.

        Parameters
        ----------
        msg_light: Union[
            messaging.MultiModelTrainRequest,
            messaging.MultiModelEvaluationRequest,
            messaging.MultiModelFairnessQuery,
        ]
            the type of message that needs to be sent by the server with the
            `weights` field
            left to None.
        clients: Set[st]
            Name of the clients to whom the message is addressed.
        needs_weights: Dict
            Dictionary that holds the corresponding model, clients that require
            the weights,
        model_weights: Dict
            Dictionary that holds the model weights to be sent.

        Returns
        -------
        Dict[str, RequestType]
            messages object to be sent to the clients under the required
            RequestType
        """
        messages: dict[
            str,
            Union[
                messaging.MultiModelTrainRequest,
                messaging.MultiModelEvaluationRequest,
                messaging.MultiModelFairnessQuery,
            ],
        ] = {}
        for client in clients:
            if isinstance(msg_light, messaging.MultiModelTrainRequest):
                train_requests = copy_requests_with_weights(
                    msg_light.requests, client, needs_weights, model_weights
                )
                messages[client] = messaging.MultiModelTrainRequest(
                    train_requests
                )

            elif isinstance(msg_light, messaging.MultiModelEvaluationRequest):
                evaluation_requests = copy_requests_with_weights(
                    msg_light.requests, client, needs_weights, model_weights
                )
                messages[client] = messaging.MultiModelEvaluationRequest(
                    evaluation_requests
                )

            else:
                fairness_requests = copy_requests_with_weights(
                    msg_light.requests, client, needs_weights, model_weights
                )
                messages[client] = messaging.MultiModelFairnessQuery(
                    fairness_requests
                )
        return messages
