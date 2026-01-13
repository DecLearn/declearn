"""Shared utils for client sampler's unit tests."""

from typing import Dict, Set

from declearn.client_sampler import ClientSampler
from declearn.messaging import TrainReply


class FailClientSampler(ClientSampler):
    """
    Client sampler that always return an empty set when sampling clients
    """

    strategy = "fail"

    @property
    def secagg_compatible(self) -> bool:
        return True

    def _sample(self, eligible_clients: Set[str]) -> Set[str]:
        return set()

    def update(self, client_to_reply: Dict[str, TrainReply]) -> None:
        pass
