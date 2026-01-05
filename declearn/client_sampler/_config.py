from dataclasses import dataclass
from typing import Any, Dict

from declearn.client_sampler import ClientSampler
from declearn.utils import TomlConfig


@dataclass
class ClientSamplerConfig(TomlConfig):
    """
    TOML-parsable configuration containers implementation for 'ClientSampler'
    """

    strategy: str
    params: Dict[str, Any]

    def build(self) -> ClientSampler:
        """Build a 'ClientSampler' instance for the configuration."""
        return ClientSampler.from_specs(strategy=self.strategy, **self.params)
