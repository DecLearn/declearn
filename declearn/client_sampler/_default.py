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

"""`ClientSampler` implementation for default sampling (select all clients)."""

from typing import Set

from declearn.client_sampler._api import ClientSampler


class DefaultClientSampler(ClientSampler):
    """Default client sampler, selecting all provided clients."""

    strategy = "default"

    @property
    def secagg_compatible(self) -> bool:
        return True

    def cls_sample(self, eligible_clients: Set[str]) -> Set[str]:
        return eligible_clients
