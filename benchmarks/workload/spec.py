"""Dataclasses describing a fully-instantiated benchmark configuration."""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from declearn.dataset import Dataset
from declearn.main.config import FLOptimConfig, FLRunConfig
from declearn.model.api import Model
from declearn.secagg.api import SecaggConfigClient, SecaggConfigServer

__all__ = [
    "BenchmarkSpec",
    "ClientSpec",
]


@dataclass
class ClientSpec:
    """Per-client inputs needed to build a `FederatedClient`."""

    name: str
    train_data: Dataset
    valid_data: Dataset
    secagg: Optional[SecaggConfigClient] = None


@dataclass
class BenchmarkSpec:
    """Container of everything `run_benchmark` needs to launch one run."""

    server_model: Model
    optim_config: FLOptimConfig
    run_config: FLRunConfig
    network_host: str
    network_port: int
    network_protocol: str
    clients: List[ClientSpec]
    server_secagg: Optional[SecaggConfigServer] = None
    metrics: List[Any] = field(default_factory=list)
