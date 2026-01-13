"""
ClientSampler implementation for weighted sampling (clients selected according
to user-defined weights).
"""

from typing import Dict, Optional, Set

import numpy as np

from declearn.client_sampler._base import ClientSampler
from declearn.messaging import TrainReply
from declearn.model.api import Model


class WeightedClientSampler(ClientSampler):
    """
    Client sampler selecting a given number of clients among all
    at random using the distribution formed by weights attributed to
    each client by the user.

    Note : this sampler assumes that the user knows in advance the name of
    all clients that will be involved in the federated process, if the
    anticipated client set and the actual one don't match, an error will
    be raised.

    Attributes
    ----------
    n_samples: int
        Number of clients to be sampled.
    client_to_weight: Dict[str, float]
        Exhaustive mapping between each client and a weight, a higher weight
        means a highest chance (proportionaly) to be selected by the sampler.
        The weights don't need to sum to one.
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
        """
        Initialize clients common metadata, then check the consistency of
        user-provided clients w.r.t. the actual clients.

        Raises
        ------
        ValueError:
            If the actual clients set `clients` do not match the clients set
            provided by the user at the sampler construction.
        """
        super().init_clients(clients)
        # check client sets match
        provided_clients = set(self.client_to_weight.keys())
        if provided_clients != clients:
            raise ValueError(
                "Clients provided at the client sampler construction "
                f"{provided_clients} do not match actual clients {clients}."
            )

    def _sample(self, eligible_clients: Set[str]) -> Set[str]:
        """
        Back-end of the sampling method for the weighted client sampler.

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

    def update(
        self, client_to_reply: Dict[str, TrainReply], server_model: Model
    ) -> None:
        pass
