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

"""TOML-parsable configuration container for `ClientSampler`."""

from dataclasses import dataclass
from typing import Any, Dict

from declearn.client_sampler import ClientSampler, instantiate_client_sampler
from declearn.utils import TomlConfig


@dataclass
class ClientSamplerConfig(TomlConfig):
    """TOML-parsable configuration container implementation for
    `ClientSampler`.
    """

    strategy: str
    params: Dict[str, Any]

    def build(self) -> ClientSampler:
        """Build a 'ClientSampler' instance for the configuration."""
        return instantiate_client_sampler(
            strategy=self.strategy, **self.params
        )
