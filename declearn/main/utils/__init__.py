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

"""Utils for the main federated learning traning and evaluation processes."""

from ._checkpoint import Checkpointer
from ._constraints import Constraint, ConstraintSet, TimeoutConstraint
from ._data_info import AggregationError, aggregate_clients_data_info
from ._early_stop import EarlyStopping, EarlyStopConfig
from ._training import TrainingManager
