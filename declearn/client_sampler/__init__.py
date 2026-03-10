# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Client sampling API, implementations and utils.

A `ClientSampler` is aimed to be used by the central server during the
federated process to select a subset of clients to participate in a federated
round (e.g. a training round). Thus, each subclass of `ClientSampler` is
characterized by its client selection strategy.

API tools
---------

* [ClientSampler][declearn.client_sampler.ClientSampler]:
    Abstract base class defining an API for client sampler.
* [ClientSamplerConfig][declearn.client_sampler.ClientSamplerConfig]:
    TOML-parsable configuration container implementation for ClientSampler.
* [list_client_samplers][declearn.client_sampler.list_client_samplers]:
    Return a mapping of registered ClientSampler subclasses.
* [instantiate_client_sampler]\
[declearn.client_sampler.instantiate_client_sampler]:
    Instantiate a `ClientSampler` from its specifications.


Concrete classes
----------------

* [CompositionClientSampler][declearn.client_sampler.CompositionClientSampler]:
    ClientSampler subclass to perform composition of several client samplers.
* [CriterionClientSampler][declearn.client_sampler.CriterionClientSampler]:
    ClientSampler subclass performing selection based on a criterion derived
    from client replies and global model.
* [DefaultClientSampler][declearn.client_sampler.DefaultClientSampler]:
    Default ClientSampler subclass selecting all clients.
* [UniformClientSampler][declearn.client_sampler.UniformClientSampler]:
    ClientSampler subclass performing selection based on uniform probability.
* [WeightedClientSampler][declearn.client_sampler.WeightedClientSampler]:
    ClientSampler subclass performing selection based on user-provided weights.
"""

from ._api import (
    ClientSampler,
    instantiate_client_sampler,
    list_client_samplers,
)
from ._composition import CompositionClientSampler
from ._config import ClientSamplerConfig
from ._criterion_sampler import CriterionClientSampler
from ._default import DefaultClientSampler
from ._uniform import UniformClientSampler
from ._weighted import WeightedClientSampler

__all__ = [
    "ClientSampler",
    "ClientSamplerConfig",
    "CompositionClientSampler",
    "CriterionClientSampler",
    "DefaultClientSampler",
    "instantiate_client_sampler",
    "UniformClientSampler",
    "WeightedClientSampler",
    "list_client_samplers",
]
