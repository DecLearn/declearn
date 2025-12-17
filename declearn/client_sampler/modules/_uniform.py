"""ClientSampler implementation for uniform sampling."""

from typing import Dict, Optional, Set

import numpy as np

from ...messaging import TrainReply
from .._base import ClientSampler


class UniformClientSampler(ClientSampler):
    """
    Client sampler selecting a given number of clients among all
    at random with uniform probability.
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

        Parameters
        ----------
        n_samples:
            Number of clients to be sampled, must be less than the
            total number of clients.
        seed:
            Optional random state used for sampling, default to None.
        """
        super().__init__(max_retries=max_retries)
        self.n_samples = n_samples
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def secagg_compatible(self) -> bool:
        return True

    def _sample(self, eligible_clients: Set[str]) -> Set[str]:
        """
        TODO doc, precise that we sample min(n_samples, len(eligible_clients)) clients
        """
        clients_list = list(eligible_clients)
        n_samples = min(self.n_samples, len(clients_list))
        sampled = self._rng.choice(clients_list, size=n_samples, replace=False)
        return {str(client_np) for client_np in sampled}

    def update(self, client_to_reply: Dict[str, TrainReply]) -> None:
        pass
