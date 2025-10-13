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

"""Tools to specify hyper-parameters of a Federated Learning process.

This submodule exposes dataclasses that group, document and facilitate
parsing (from instances, config dicts and/or TOML files) elements that
are required to specify a Federated Learning process from the server's
side.

The main classes implemented here are:

* [FLRunConfig][declearn.main.config.FLRunConfig]:
    Federated learning orchestration hyper-parameters.
* [FLOptimConfig][declearn.main.config.FLOptimConfig]:
    Federated optimization strategy.

The following dataclasses are articulated by `FLRunConfig`:

* [EvaluateConfig][declearn.main.config.EvaluateConfig]:
    Hyper-parameters for an evaluation round.
* [FairnessConfig][declearn.main.config.FairnessConfig]:
    Dataclass wrapping parameters for fairness evaluation rounds.
* [RegisterConfig][declearn.main.config.RegisterConfig]:
    Hyper-parameters for clients registration.
* [TrainingConfig][declearn.main.config.TrainingConfig]:
    Hyper-parameters for a training round.
"""

from ._dataclasses import (
    EvaluateConfig,
    FairnessConfig,
    PrivacyConfig,
    RegisterConfig,
    TrainingConfig,
)
from ._run_config import FLRunConfig
from ._strategy import FLOptimConfig
