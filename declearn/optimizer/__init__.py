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

"""Framework-agnostic optimizer tools, both generic or FL-specific.

In more details, we here define an `Optimizer` class that wraps together a set
of plug-in modules, used to implement various optimization and regularization
techniques.

Main class:

* [Optimizer][declearn.optimizer.Optimizer]:
    Base class to define gradient-descent-based optimizers.

Submodules providing with plug-in algorithms:

* [modules][declearn.optimizer.modules]:
    Gradients-alteration algorithms, implemented as plug-in modules.
* [regularizers][declearn.optimizer.regularizers]:
    Loss-regularization algorithms, implemented as plug-in modules.
* [schedulers][declearn.optimizer.schedulers]:
    Time-based schedulers for learning rate or weight decay.

Utils to list available plug-ins:

* [list_optim_modules][declearn.optimizer.list_optim_modules]:
    Return a mapping of registered OptiModule subclasses.
* [list_optim_regularizers][declearn.optimizer.list_optim_regularizers]:
    Return a mapping of registered Regularizer subclasses.
* [list_rate_schedulers][declearn.optimizer.list_rate_schedulers]:
    Return a mapping of registered Scheduler subclasses.
"""

from . import modules, regularizers
from ._base import Optimizer
from ._utils import (
    list_optim_modules,
    list_optim_regularizers,
    list_rate_schedulers,
)
