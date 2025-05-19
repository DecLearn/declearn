import random
from typing import Dict, Set

from ...messaging import Message
from .._base import ClientSampler


class UniformClientSampler(ClientSampler):
    """
    Client sampler taking clients at random with uniform probability.
    """
    secagg_compatible = True

    def cls_sample(self, input_clients: Set[str]) -> Set[str]:
        return set(
            random.choices(
                list(input_clients),
                weights=self.prior_weights,
                k=self.n_samples,
            )
        )

    def update(self, results: Dict[str, Message]):
        pass
