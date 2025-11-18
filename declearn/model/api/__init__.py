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

"""Model and Vector abstractions submodule.

This submodules exports the building blocks of the Model and Vector APIs:

* [Model][declearn.model.api.Model]:
    Abstract class defining an API to interface a ML model.
* [Vector][declearn.model.api.Vector]:
    Abstract class defining an API to manipulate (sets of) data arrays.
* [VectorSpec][declearn.model.api.VectorSpec]:
    Metadata container to specify a Vector for its (un)flattening.
* [register_vector_type][declearn.model.api.register_vector_type]:
    Decorate a Vector subclass to make it buildable with `Vector.build`.
"""

from ._model import Model
from ._vector import Vector, VectorSpec, register_vector_type
