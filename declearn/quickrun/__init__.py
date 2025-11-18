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

"""Script to quickly run a simulated FL example locally using declearn.

This submodule, which is not imported by default, mainly aims at providing
with the `declearn-quickrun` command-line entry-point so as to easily set
up and run simulated federated learning experiments on a single computer.

It exposes the following, merely as a way to make the documentation of that
util available to end-users:

- [quickrun][declearn.quickrun.quickrun]:
    Backend function of the `declearn-quickrun` command-line entry-point.
- [parse_data_folder][declearn.quickrun.parse_data_folder]:
    Util to parse through a data folder used in a quickrun experiment.
- [DataSourceConfig][declearn.quickrun.DataSourceConfig]:
    Dataclass and TOML parser for data-parsing hyper-parameters.
- [ExperimentConfig][declearn.quickrun.ExperimentConfig]:
    Dataclass and TOML parser for experiment-defining hyper-parameters.
- [ModelConfig][declearn.quickrun.ModelConfig]:
    Dataclass and TOML parser for model-defining hyper-parameters.
"""

from ._config import DataSourceConfig, ExperimentConfig, ModelConfig
from ._parser import parse_data_folder
from ._run import quickrun
