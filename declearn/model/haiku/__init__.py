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

"""Haiku models interfacing tools.

Haiku is a Google DeepMind library that provides with tools to build
artificial neural networks backed by the JAX computation library. We
selected it as a primary candidate to support using JAX-backed models,
mostly because of its simplicity, that leaves apart some components
that DecLearn already provides (such as optimization algorithms).

In July 2023, Haiku development was announced to be stalled as far as
new features are concerned, in favor of Flax, a concurrent Google project.

DecLearn is planned to add support for Flax at some point (building on the
existing Haiku-oriented code, notably as far as Jax NumPy is concerned).
In the meanwhile, this submodule enables running code that operates using
haiku, which probably does not cover a lot of use cases, but it bound to
keep working at least for a while, until Google decides to drop maintenance
altogether.

This module exposes:
* HaikuModel: Model subclass to wrap haiku.Model objects
* JaxNumpyVector: Vector subclass to wrap jax.numpy.ndarray objects
"""

from . import utils
from ._model import HaikuModel
from ._vector import JaxNumpyVector
