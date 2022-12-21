# coding: utf-8

"""TOML-parsable container for Federated Learning "run" configurations."""

import dataclasses
from typing import Any, Optional


from declearn.main.utils import EarlyStopConfig
from declearn.main.config._dataclasses import (
    EvaluateConfig,
    TrainingConfig,
    RegisterConfig,
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
    early_stop: EarlyStopConfig or None, default=None
        Optional parameters to set up an EarlyStopping criterion, to
        be leveraged so as to interrupt the federated learning process
        based on the tracking of a minimized quantity (e.g. model loss).

    Instantiation classmethods
    --------------------------
    from_toml:
        Instantiate by parsing a TOML configuration file.
    from_params:
        Instantiate by parsing inputs dicts (or objects).
    """

    rounds: int
    register: RegisterConfig
    training: TrainingConfig
    evaluate: EvaluateConfig
    early_stop: Optional[EarlyStopConfig] = None  # type: ignore  # is a type

    @classmethod
    def parse_register(
        cls,
        field: dataclasses.Field[RegisterConfig],
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
            return RegisterConfig(min_clients=1)
        return cls.default_parser(field, inputs)

    @classmethod
    def from_params(
        cls,
        **kwargs: Any,
    ) -> "FLRunConfig":
        # If evaluation batch size is not set, use the same as training.
        # Note: if inputs have invalid formats, let the parent method fail.
        evaluate = kwargs.setdefault("evaluate", {})
        if isinstance(evaluate, dict):
            training = kwargs.get("training")
            if isinstance(training, dict):
                evaluate.setdefault("batch_size", training.get("batch_size"))
            elif isinstance(training, TrainingConfig):
                evaluate.setdefault("batch_size", training.batch_size)
        # Delegate the rest of the work to the parent method.
        return super().from_params(**kwargs)
