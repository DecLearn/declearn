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

"""Declearn - a python package for private decentralized learning.

Declearn is a modular framework to set up and run federated learning
processes. It is being developed by the MAGNET team of INRIA Lille,
with the aim of providing users with a modular and extensible framework
to implement federated learning algorithms and apply them to real-world
(or simulated) data using any common machine learning framework.

Declearn provides with abstractions that enable algorithms to be written
agnostic to the actual computation framework as well as with workable
interfaces that cover some of the most popular frameworks, such as
Scikit-Learn, TensorFlow and PyTorch.

The package is organized into the following submodules:
* aggregator:
    Model updates aggregating API and implementations.
* communication:
    Client-Server network communications API and implementations.
* data_info:
    Tools to write and extend shareable metadata fields specifications.
* dataset:
    Data interfacing API and implementations.
* main:
    Main classes implementing a Federated Learning process.
* metrics:
    Iterative and federative evaluation metrics computation tools.
* model:
    Model interfacing API and implementations.
* optimizer:
    Framework-agnostic optimizer and algorithmic plug-ins API and tools.
* typing:
    Type hinting utils, defined and exposed for code readability purposes.
* utils:
    Shared utils used (extensively) across all of declearn.
"""

from . import (
    aggregator,
    communication,
    data_info,
    dataset,
    main,
    metrics,
    model,
    optimizer,
    typing,
    utils,
)

__version__ = "2.1.0"
