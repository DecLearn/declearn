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

"""Shared utils for client sampler's testing."""

from typing import Set

from declearn.client_sampler import ClientSampler


class FailClientSampler(ClientSampler):
    """
    Client sampler that always return an empty set when sampling clients
    """

    strategy = "fail"

    @property
    def secagg_compatible(self) -> bool:
        return True

    def cls_sample(self, eligible_clients: Set[str]) -> Set[str]:
        return set()
