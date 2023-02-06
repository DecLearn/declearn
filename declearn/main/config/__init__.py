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
