"""Workload-construction layer for the declearn benchmark suite.

This subpackage assembles a `BenchmarkSpec` from a small set of
parameters and provides a runner that turns a spec into a live
federated learning experiment using `FederatedServer` / `FederatedClient`.
"""

from benchmarks.workload.build import build_benchmark
from benchmarks.workload.runner import run_benchmark
from benchmarks.workload.spec import BenchmarkSpec, ClientSpec

__all__ = [
    "BenchmarkSpec",
    "ClientSpec",
    "build_benchmark",
    "run_benchmark",
]
