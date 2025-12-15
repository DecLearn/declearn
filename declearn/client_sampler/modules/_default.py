"""ClientSampler implementation for default sampling (select all clients)."""

from typing import Dict, Set

from declearn.client_sampler._base import ClientSampler
from declearn.messaging import TrainReply


class DefaultClientSampler(ClientSampler):
    """
    Default client sampler which actually don't sample, because
    it selects all provided clients.
    """

    strategy = "default"

    @property
    def secagg_compatible(self) -> bool:
        return True

    def _sample(self, eligible_clients: Set[str]) -> Set[str]:
        return eligible_clients

    def update(self, client_to_reply: Dict[str, TrainReply]) -> None:
        pass
