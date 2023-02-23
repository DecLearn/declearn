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

"""Model interfacing submodule, defining an API an derived applications.

This declearn submodule provides with:
* Model and Vector abstractions, used as an API to design FL algorithms
* Submodules implementing interfaces to various frameworks and models.

The automatically-imported submodules implemented here are:
* api: Model and Vector abstractions' defining module.
    - Model: abstract API to interface framework-specific models.
    - Vector: abstract API for data tensors containers.
* sklearn: scikit-learn based or oriented tools
    - NumpyVector: Vector for numpy array data structures.
    - SklearnSGDModel: Model for scikit-learn's SGDClassifier and SGDRegressor.

The optional-dependency-based submodules that may be manually imported are:
* tensorflow: tensorflow-interfacing tools
    - TensorflowModel: Model to wrap any tensorflow-keras Layer model.
    - TensorflowOptiModule: Hacky OptiModule to wrap a keras Optimizer.
    - TensorflowVector: Vector for tensorflow Tensor and IndexedSlices.
* torch: pytorch-interfacing tools
    - TorchModel: Model to wrap any torch Module model.
    - TorchOptiModule: Hacky OptiModule to wrap a torch Optimizer.
    - TorchVector: Vector for torch Tensor objects.
"""

from . import api
from . import sklearn

OPTIONAL_MODULES = [
    "tensorflow",
    "torch",
]
