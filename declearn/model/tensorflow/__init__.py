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

"""Tensorflow models interfacing tools.

This submodule provides with a generic interface to wrap up
any TensorFlow `keras.Model` instance that is to be trained
through gradient descent.

This module exposes:
* TensorflowModel: Model subclass to wrap tensorflow.keras.Model objects
* TensorflowVector: Vector subclass to wrap tensorflow.Tensor objects
"""

from ._vector import TensorflowVector
from ._model import TensorflowModel
