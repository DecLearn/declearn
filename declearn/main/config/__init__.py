# coding: utf-8

"""Tools to specify hyper-parameters of a Federated Learning process.

This submodule exposes dataclasses that group and document server-side
hyper-parameters that specify a Federated Learning process, as well as
a main class designed to act as a container and a parser for all these,
that may be instantiated from python objects or from a TOML file.

In other words, `FLRunConfig` in the key class implemented here, while
the other exposed dataclasses are already articulated and used by it.
"""

from ._dataclasses import (
    EvaluateConfig,
    RegisterConfig,
    TrainingConfig,
)
from ._run_config import FLRunConfig
