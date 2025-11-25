import random
from typing import Dict, Optional, Set

from ...messaging import Message
from .._base import ClientSampler


class UniformClientSampler(ClientSampler):
    """
    Client sampler selecting clients at random with uniform probability.

    If the 'prior_weights' attribute is set, uses the weights to alter
    sampling probabilities instead of using the uniform distribution.
    """

    secagg_compatible = True

    def __init__(
        self,
        n_samples: int,
        clients: Optional[Set[str]] = None,
        prior_weights: Optional[Dict[str, float]] = None,
        initialization_round: bool = False,
    ):
        """
        TODO
        """
        super().__init__(clients, prior_weights, initialization_round)
        self.n_samples = n_samples

    def _sample(self, input_clients: Set[str]) -> Set[str]:
        clients_list = list(input_clients)
        if self.prior_weights is None:
            weights = None
        else:
            weights = [self.prior_weights[client] for client in clients_list]
        return set(
            random.choices(
                clients_list,
                weights=weights,
                k=self.n_samples,
            )
        )

    def update(self, results: Dict[str, Message]) -> None:
        pass
