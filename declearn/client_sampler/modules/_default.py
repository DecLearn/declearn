"""ClientSampler implementation for default sampling (select all clients)."""

from typing import Dict, Set

from declearn.client_sampler._base import ClientSampler
from declearn.messaging import TrainReply


class DefaultClientSampler(ClientSampler):
    """
    Default client sampler which actually don't sample, because
    it selects all clients.
    """

    name = "default"

    @property
    def secagg_compatible(self) -> bool:
        return True

    def cls_sample(self) -> Set[str]:
        return self.clients

    def update(self, client_to_reply: Dict[str, TrainReply]) -> None:
        pass
