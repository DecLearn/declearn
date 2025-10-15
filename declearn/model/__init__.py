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

"""Model interfacing submodule, defining an API an derived applications.

This declearn submodule provides with:

- Model and Vector abstractions, used as an API to design FL algorithms.
- Submodules implementing interfaces to various frameworks and models.

Default Submodules
------------------
The automatically-imported submodules implemented here are:

* [api][declearn.model.api]:
    Model and Vector abstractions' defining module.
    - [Model][declearn.model.api.Model]:
        abstract API to interface framework-specific models.
    - [Vector][declearn.model.api.Vector]:
        abstract API for data tensors containers.
* [sklearn][declearn.model.sklearn]:
    Scikit-Learn based or oriented tools
    - [NumpyVector][declearn.model.sklearn.NumpyVector]
        Vector for numpy array data structures.
    - [SklearnSGDModel][declearn.model.sklearn.SklearnSGDModel]
        Model for scikit-learn's SGDClassifier and SGDRegressor.

Optional Submodules
-------------------
The optional-dependency-based submodules that may be manually imported are:

* [haiku][declearn.model.haiku]:
    Jax- and Haiku-interfacing tools.
    - [HaikuModel][declearn.model.haiku.HaikuModel]:
        Model to wrap a haiku-transformable model function.
    - [JaxNumpyVector][declearn.model.haiku.JaxNumpyVector]:
        Vector for jax array data structures.
* [tensorflow][declearn.model.tensorflow]:
    TensorFlow-interfacing tools
    - [TensorflowModel][declearn.model.tensorflow.TensorflowModel]:
        Model to wrap any tensorflow-keras Layer model.
    - [TensorflowOptiModule][declearn.model.tensorflow.TensorflowOptiModule]:
        Hacky OptiModule to wrap a keras Optimizer.
    - [TensorflowVector][declearn.model.tensorflow.TensorflowVector]:
        Vector for tensorflow Tensor and IndexedSlices.
* [torch][declearn.model.torch]:
    PyTorch-interfacing tools
    - [TorchModel][declearn.model.torch.TorchModel]:
        Model to wrap any torch Module model.
    - [TorchOptiModule][declearn.model.torch.TorchOptiModule]:
        Hacky OptiModule to wrap a torch Optimizer.
    - [TorchVector][declearn.model.torch.TorchVector]:
        Vector for torch Tensor objects.
"""

from . import api, sklearn

OPTIONAL_MODULES = [
    "haiku",
    "tensorflow",
    "torch",
]
