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

"""Pytorch models interfacing tools.

This submodule provides with a generic interface to wrap up any PyTorch
`torch.nn.Module` instance that is to be trained with gradient descent.

This module exposes the following classes:

* [TorchModel][declearn.model.torch.TorchModel]:
    Model subclass to wrap torch.nn.Module objects.
* [TorchOptiModule][declearn.model.torch.TorchOptiModule]:
    OptiModule subclass to wrap torch.nn.Optimizer objects.
* [TorchVector][declearn.model.torch.TorchVector]:
    Vector subclass to wrap torch.Tensor objects.

It also exposes the [utils][declearn.model.torch.utils] submodule, which
mainly aims at providing tools used in the backend of the former objects.
"""

from . import utils
from ._model import TorchModel
from ._optim import TorchOptiModule
from ._vector import TorchVector
