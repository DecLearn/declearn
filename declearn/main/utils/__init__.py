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

"""Utils for the main federated learning traning and evaluation processes.

End-user utils
--------------
Utils that may be composed into the main orchestration classes:

* [Checkpointer][declearn.main.utils.Checkpointer]:
    Model, optimizer, and metrics checkpointing class.
* [EarlyStopping][declearn.main.utils.EarlyStopping]:
    Class implementing a metric-based early-stopping decision rule.
* [EarlyStopConfig][declearn.main.utils.EarlyStopConfig]:
    Dataclass for EarlyStopping instantiation parameters.

Backend: data-info aggregation
------------------------------
Backend utils to aggregate clients' dataset information:

* [aggregate_clients_data_info]\
[declearn.main.utils.aggregate_clients_data_info]:
    Validate and aggregate clients' data-info dictionaries.
* [AggregationError][declearn.main.utils.AggregationError]:
    Custom exception that may be raised by `aggregate_clients_data_info`.


DEPRECATED TrainingManager
--------------------------
This class has been moved to `declearn.training.TrainingManager` as of
DecLearn 2.6. It is re-exported merely for retro-compatibility purposes,
but this import path will be removed in DecLearn 2.8.

* [TrainingManager][declearn.training.TrainingManager]:
    Class wrapping the logic for local training and evaluation rounds.


DEPRECATED Backend: effort constraints
--------------------------------------

Backend utils that are used to specify and articulate effort constraints
for training and evaluation rounds:

* [Constraint][declearn.main.utils.Constraint]:
    Base class to implement effort constraints.
* [ConstraintSet][declearn.main.utils.ConstraintSet]:
    Utility class to wrap sets of Constraint instances.
* [TimeoutConstraint][declearn.main.utils.TimeoutConstraint]:
    Class implementing a simple time-based constraint.

The following components have been moved elsewhere and made private as of
DecLearn 2.6. They are re-exported from this module for retro-compatibility
but will be removed in DecLearn 2.8.
**If you are using them, let us know so that we may amend this decision.**
"""

# Deprecated re-exports. FUTURE: remove these (DecLearn >=2.8)
from declearn.training import TrainingManager
from declearn.training._constraints import (
    Constraint,
    ConstraintSet,
    TimeoutConstraint,
)

from ._checkpoint import Checkpointer
from ._data_info import AggregationError, aggregate_clients_data_info
from ._early_stop import EarlyStopConfig, EarlyStopping
