from dataclasses import dataclass
from typing import Any, Dict

from declearn.client_sampler import ClientSampler
from declearn.utils import TomlConfig, access_registered


@dataclass
class ClientSamplerConfig(TomlConfig):
    """
    TODO doc
    """

    strategy: str
    params: Dict[str, Any]

    def build(self) -> ClientSampler:
        cls = access_registered(self.strategy, group="ClientSampler")
        return cls(**self.params)
