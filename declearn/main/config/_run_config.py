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

"""TOML-parsable container for Federated Learning "run" configurations."""

import dataclasses
from typing import Any, Optional

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.main.utils import EarlyStopConfig
from declearn.main.config._dataclasses import (
    EvaluateConfig,
    PrivacyConfig,
    RegisterConfig,
    TrainingConfig,
)
from declearn.utils import TomlConfig


__all__ = [
    "FLRunConfig",
]


@dataclasses.dataclass
class FLRunConfig(TomlConfig):
    """Global container for Federated Learning "run" configurations.

    This class aims at wrapping multiple, possibly optional, sets of
    hyper-parameters that parameterize a Federated Learning process,
    each of which is specified through a dedicated dataclass or as a
    unit python type.

    It is designed to be use by an orchestrator, e.g. the server in
    the case of a centralized federated learning process.

    This class is meant to be extendable through inheritance, so as
    to refine the expected fields or add some that might be used by
    children (or parallel) classes of `FederatedServer` that modify
    the default, centralized, federated learning process.

    Fields
    ------
    rounds: int
        Maximum number of training and validation rounds to perform.
    register: RegisterConfig
        Parameters for clients' registration (min and/or max number
        of clients to expect, optional max duration of the process).
    training: TrainingConfig
        Parameters for training rounds, including effort constraints
        and data-batching instructions.
    evaluate: EvaluateConfig
        Parameters for validation rounds, similar to training ones.
    privacy: PrivacyConfig or None
        Optional parameters to set up local differential privacy,
        by having clients use the DP-SGD algorithm for training.
    early_stop: EarlyStopConfig or None
        Optional parameters to set up an EarlyStopping criterion, to
        be leveraged so as to interrupt the federated learning process
        based on the tracking of a minimized quantity (e.g. model loss).

    Instantiation classmethods
    --------------------------
    from_toml:
        Instantiate by parsing a TOML configuration file.
    from_params:
        Instantiate by parsing inputs dicts (or objects).

    Notes
    -----
    * `register` may be defined as a single integer (in `from_params` or in
      a TOML file), that will be used as the exact number of clients.
    * If `evaluate` is not provided to `from_params` or in the parsed TOML
      file, default parameters will automatically be used and the training
      batch size will be used for evaluation as well.
    * If `privacy` is provided and the 'poisson' parameter is unspecified
      for `training`, it will be set to True by default rather than False.
    """

    rounds: int
    register: RegisterConfig
    training: TrainingConfig
    evaluate: EvaluateConfig
    privacy: Optional[PrivacyConfig] = None
    early_stop: Optional[EarlyStopConfig] = None  # type: ignore  # is a type

    @classmethod
    def parse_register(
        cls,
        field: dataclasses.Field,  # future: dataclasses.Field[RegisterConfig]
        inputs: Any,
    ) -> RegisterConfig:
        """Field-specific parser to instantiate a RegisterConfig.

        This method supports specifying `register`:
        * as a single int, translated into {"min_clients": inputs}
        * as None (or missing kwarg), using default RegisterConfig()

        It otherwise routes inputs back to the `default_parser`.
        """
        if inputs is None:
            return RegisterConfig()
        if isinstance(inputs, int):
            return RegisterConfig(min_clients=inputs)
        return cls.default_parser(field, inputs)

    @classmethod
    def from_params(
        cls,
        **kwargs: Any,
    ) -> Self:
        # If evaluation batch size is not set, use the same as training.
        # Note: if inputs have invalid formats, let the parent method fail.
        evaluate = kwargs.setdefault("evaluate", {})
        if isinstance(evaluate, dict):
            training = kwargs.get("training")
            if isinstance(training, dict):
                evaluate.setdefault("batch_size", training.get("batch_size"))
            elif isinstance(training, TrainingConfig):
                evaluate.setdefault("batch_size", training.batch_size)
        # If privacy is set and poisson sampling bool parameter is unspecified
        # for the training dataset, make it True rather than False by default.
        privacy = kwargs.get("privacy")
        if isinstance(privacy, dict):
            training = kwargs.get("training")
            if isinstance(training, dict):
                training.setdefault("poisson", True)
        # Delegate the rest of the work to the parent method.
        return super().from_params(**kwargs)
