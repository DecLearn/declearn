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

"""Framework-agnostic optimizer tools, both generic or FL-specific.

In more details, we here define an `Optimizer` class that wraps together a set
of plug-in modules, used to implement various optimization and regularization
techniques.

Main class:
* Optimizer: Base class to define gradient-descent-based optimizers.

This module also implements the following submodules, used by the former:
* modules: gradients-alteration algorithms, implemented as plug-in modules.
* regularizers: loss-regularization algorithms, implemented as plug-in modules.

 """


from . import modules, regularizers
from ._base import Optimizer
