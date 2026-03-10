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

"""
`ClientSampler` implementation for weighted sampling, i.e. client selection
based on user-defined weights.
"""

from typing import Dict, Optional, Set

import numpy as np

from declearn.client_sampler._api import ClientSampler


class WeightedClientSampler(ClientSampler):
    """Client sampler selecting randomly a given number of clients among all
    using user-defined per-client weights.

    Notes
    -----
    This sampler assumes that the user knows in advance the name of
    all clients that will be involved in the federated process. If the
    anticipated clients set is not a superset of the actual (= registered) one,
    an error will be raised.

    Attributes
    ----------
    n_samples: int
        Number of clients to be sampled.
    client_to_weight: Dict[str, float]
        Exhaustive mapping between each client and a weight.
        A higher weight means a higher chance (proportionaly) to be selected by
        the sampler. The weights don't need to sum to one.
    seed: Optional[int]
        Optional random state used for sampling.
    """

    strategy = "weighted"

    def __init__(
        self,
        n_samples: int,
        client_to_weight: Dict[str, float],
        seed: Optional[int] = None,
        max_retries: int = ClientSampler.DEFAULT_MAX_RETRIES,
    ):
        """
        Instantiate the weighted client sampler.
        """
        super().__init__(max_retries=max_retries)
        self.n_samples = n_samples
        self.client_to_weight = client_to_weight
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def secagg_compatible(self) -> bool:
        return True

    def init_clients(self, clients: Set[str]) -> None:
        """Initialize clients common metadata.

        Check the consistency of user-provided clients w.r.t. the actual
        clients, i.e. clients provided at the sampler construction must be a
        superset of actual clients (this method parameter).

        Raises
        ------
        ValueError:
            If the anticipated clients set is not a superset of the actual
            clients.
        """
        super().init_clients(clients)
        # check client sets match
        anticipated_clients = set(self.client_to_weight.keys())
        if not anticipated_clients.issuperset(clients):
            raise ValueError(
                "Clients set provided at the client sampler construction "
                f"{anticipated_clients} do not contain all actual clients "
                f"{clients}."
            )

    def cls_sample(self, eligible_clients: Set[str]) -> Set[str]:
        """Back-end of the sampling method for the weighted client sampler.

        If there are more than `n_samples` clients in `eligible_clients`, this
        method samples this number of clients with probability computed from
        the user-provided weights, without replacement.
        Otherwise, they are all selected.
        """
        if self.n_samples >= len(eligible_clients):
            return eligible_clients

        clients_list = list(eligible_clients)
        weights_sum = sum(
            [self.client_to_weight[client] for client in clients_list]
        )
        probas = [
            self.client_to_weight[client] / weights_sum
            for client in clients_list
        ]
        n_samples = min(self.n_samples, len(clients_list))
        sampled = self._rng.choice(
            clients_list, size=n_samples, replace=False, p=probas
        )
        return {str(client_np) for client_np in sampled}
