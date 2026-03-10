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

"""`ClientSampler` implementation for uniform sampling."""

from typing import Optional, Set

import numpy as np

from declearn.client_sampler._api import ClientSampler


class UniformClientSampler(ClientSampler):
    """Client sampler selecting randomly a given number of clients among all
    with uniform probability.

    Attributes
    ----------
    n_samples: int
        Number of clients to be sampled.
    seed: Optional[int]
        Optional random state used for sampling.
    """

    strategy = "uniform"

    def __init__(
        self,
        n_samples: int,
        seed: Optional[int] = None,
        max_retries: int = ClientSampler.DEFAULT_MAX_RETRIES,
    ):
        """
        Instantiate the uniform client sampler.
        """
        super().__init__(max_retries=max_retries)
        self.n_samples = n_samples
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def secagg_compatible(self) -> bool:
        return True

    def cls_sample(self, eligible_clients: Set[str]) -> Set[str]:
        """
        Back-end of the sampling method for the uniform client sampler.

        If there are more than `n_samples` clients in `eligible_clients`, this
        method samples this number of clients with uniform probability, without
        replacement. Otherwise, they are all selected.
        """
        if self.n_samples >= len(eligible_clients):
            return eligible_clients

        clients_list = list(eligible_clients)
        n_samples = min(self.n_samples, len(clients_list))
        sampled = self._rng.choice(clients_list, size=n_samples, replace=False)
        return {str(client_np) for client_np in sampled}
