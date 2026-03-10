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

"""`ClientSampler` implementation for composition of client samplers."""

from typing import Any, Dict, List, Set

from declearn.client_sampler._api import (
    ClientSampler,
    instantiate_client_sampler,
)
from declearn.messaging import TrainReply
from declearn.model.api import Model


class CompositionClientSampler(ClientSampler):
    """Client sampler that composes a list of other samplers sequentially.

    The composition mechanism works the following way: the first sampler
    selects some client(s), then the second one selects other(s) among the
    remaining ones, and so on.
    At the end, the clients selected by the `CompositionClientSampler` are the
    union of client sets selected by each sampler, consecutively.

    Attributes
    ----------
    samplers: List[ClientSampler]
        list of client samplers to be combined.
    """

    strategy = "composition"

    def __init__(
        self,
        samplers: List[ClientSampler],
        max_retries: int = ClientSampler.DEFAULT_MAX_RETRIES,
    ):
        super().__init__(max_retries=max_retries)
        self.samplers = samplers

    @property
    def secagg_compatible(self) -> bool:
        # Composition client sampler is secagg-compatible if all of its
        # samplers are.
        return all([sampler.secagg_compatible for sampler in self.samplers])

    def init_clients(self, clients: Set[str]) -> None:
        super().init_clients(clients)
        for sampler in self.samplers:
            sampler.init_clients(clients)

    def cls_sample(self, eligible_clients: Set[str]) -> Set[str]:
        total_sampled_clients = set()
        for sampler in self.samplers:
            sampler_clients = sampler.sample(eligible_clients)
            total_sampled_clients.update(sampler_clients)
            eligible_clients = eligible_clients - sampler_clients

        return total_sampled_clients

    def update(
        self, client_to_reply: Dict[str, TrainReply], global_model: Model
    ) -> None:
        for sampler in self.samplers:
            sampler.update(client_to_reply, global_model)

    @classmethod
    def from_specs(cls, **kwargs: Any) -> ClientSampler:
        """Instantiate a `CompositionClientSampler` from its specifications.

        Notes
        -----
        Each sampler composing the 'samplers' list can be either
        a dictionnary of valid sampler specification, or an instance of
        `ClientSampler`.
        """
        samplers = kwargs["samplers"]
        parsed_samplers = []
        for sampler in samplers:
            if isinstance(sampler, ClientSampler):
                parsed_samplers.append(sampler)
            elif isinstance(sampler, dict):
                parsed_samplers.append(instantiate_client_sampler(**sampler))
            else:
                raise ValueError(
                    f"Unsupported sampler type '{type(sampler)}' in "
                    "samplers list"
                )
        kwargs["samplers"] = parsed_samplers
        return cls(**kwargs)
