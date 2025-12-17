from dataclasses import dataclass
from typing import Any, Dict

from declearn.client_sampler import ClientSampler
from declearn.utils import TomlConfig


@dataclass
class ClientSamplerConfig(TomlConfig):
    """
    TODO doc
    """

    strategy: str
    params: Dict[str, Any]

    def build(self) -> ClientSampler:
        return ClientSampler.from_specs(strategy=self.strategy, **self.params)
