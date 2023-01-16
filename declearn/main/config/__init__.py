# coding: utf-8

"""Tools to specify hyper-parameters of a Federated Learning process.

This submodule exposes dataclasses that group, document and facilitate
parsing (from instances, config dicts and/or TOML files) elements that
are required to specify a Federated Learning process from the server's
side.

The main classes implemented here are:
* FLRunConfig   : federated learning orchestration hyper-parameters
* FLOptimConfig : federated optimization strategy

The following dataclasses are articulated by `FLRunConfig`:
* EvaluateConfig : hyper-parameters for an evaluation round
* RegisterConfig : hyper-parameters for clients registration
* TrainingConfig : hyper-parameters for a training round


This submodule exposes dataclasses that group and document server-side
hyper-parameters that specify a Federated Learning process, as well as
a main class designed to act as a container and a parser for all these,
that may be instantiated from python objects or from a TOML file.

In other words, `FLRunConfig` in the key class implemented here, while
the other exposed dataclasses are already articulated and used by it.
"""

from ._dataclasses import (
    EvaluateConfig,
    PrivacyConfig,
    RegisterConfig,
    TrainingConfig,
)
from ._run_config import FLRunConfig
from ._strategy import FLOptimConfig
