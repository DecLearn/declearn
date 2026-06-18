"""Fixed baseline configuration for the declearn benchmark suite.

Every benchmark category varies exactly one axis from the values
defined here. Anything not exposed via `build_benchmark`'s kwargs
must come from this module so that benchmarks remain comparable
across categories and across declearn versions.
"""

from typing import Any, List

__all__ = [
    "BASELINE_AGGREGATOR",
    "BASELINE_BACKEND",
    "BASELINE_BATCH_SIZE",
    "BASELINE_CLIENT_LRATE",
    "BASELINE_CLIENT_MODULES",
    "BASELINE_DATASET_FRACTION",
    "BASELINE_EVAL_BATCH_SIZE",
    "BASELINE_METRICS",
    "BASELINE_NETWORK_HOST",
    "BASELINE_NETWORK_PORT",
    "BASELINE_NETWORK_PROTOCOL",
    "BASELINE_N_CLIENTS",
    "BASELINE_REGISTRATION_TIMEOUT",
    "BASELINE_ROUNDS",
    "BASELINE_SERVER_LRATE",
]


BASELINE_BACKEND: str = "torch"
BASELINE_AGGREGATOR: str = "averaging"
BASELINE_CLIENT_LRATE: float = 0.001
BASELINE_CLIENT_MODULES: List[str] = ["adam"]
BASELINE_SERVER_LRATE: float = 1.0
BASELINE_ROUNDS: int = 2
BASELINE_BATCH_SIZE: int = 48
BASELINE_EVAL_BATCH_SIZE: int = 128
BASELINE_N_CLIENTS: int = 3
BASELINE_REGISTRATION_TIMEOUT: int = 60
# Fraction of each client's MNIST shard used for benchmarking. The full
# FL pipeline (registration, aggregation, eval, encryption) is exercised
# identically; only the inner training loop sees less data.
# This trades MNIST sample count — irrelevant to a "did declearn's plumbing
# get slower" benchmark — for a 5–10× per-cell speedup. Set to 1.0 to
# restore the full split; cached layout dirs are keyed on this value.
BASELINE_DATASET_FRACTION: float = 0.1
BASELINE_NETWORK_HOST: str = "127.0.0.1"
BASELINE_NETWORK_PORT: int = 8765
BASELINE_NETWORK_PROTOCOL: str = "websockets"
BASELINE_METRICS: List[Any] = [
    ["multi-classif", {"labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]
]
