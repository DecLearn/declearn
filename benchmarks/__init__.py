"""ASV benchmark classes for declearn.

Each class is a thin wrapper over `build_benchmark(...)` +
`run_benchmark(...)`. ASV discovers them automatically. Heavy lifting
(parameter interpretation, data preparation, network/model setup) lives
in `benchmarks.workload`. The classes here only declare which slice of
the parameter space each benchmark exercises.

The `n_clients` axis defaults to a single point (`[5]`) to keep the
per-version sweep under ~2 hours. The active axis is sourced from the
`DECLEARN_BENCH_N_CLIENTS` env var (comma-separated positive ints,
e.g. "5,20") when set, so that `bench.yaml` profiles can widen or
narrow the sweep without editing this file. Every class — including
SecAggBenchmark — honors the same axis. The CI regression-check profile
(`ci`) runs all four classes but keeps the axis at n=5, so the slow
cells (the n=20 masking cell historically hit `SecAggBenchmark.timeout`)
are only paid for in manual `scale` runs where they are visible to the
operator.

Memory tracking: `time_run` measures peak host-RSS-delta and peak
GPU bytes as side effects (microseconds; invisible against the
~70-180s workload) and writes both to a `/tmp` cache. The
`track_peakmem_run` and `track_peakgpu_run` methods only read from
that cache — no extra workload runs, so the sweep stays the same
length. The delta-RSS form sidesteps the "setup dominates peak"
blind spot; the GPU number sidesteps "ru_maxrss misses VRAM".
"""

import json
import os
import resource
import tempfile
from typing import List, Tuple

import torch

from benchmarks.workload import build_benchmark, run_benchmark
from benchmarks.workload.build import BACKEND_LAYOUT
from benchmarks.workload.data import ensure_data_for_n_clients


def _cuda() -> bool:
    return torch.cuda.is_available()


def _cache_path(cls_name: str, params: Tuple) -> str:
    key = "_".join(str(p) for p in params)
    return os.path.join(
        tempfile.gettempdir(), f"declearn_mem_{cls_name}_{key}.json"
    )


def _mem_capture_start() -> int:
    if _cuda():
        torch.cuda.reset_peak_memory_stats()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _mem_capture_end(cls_name: str, params: Tuple, baseline_kb: int) -> None:
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    host_delta_bytes = max(peak_kb - baseline_kb, 0) * 1024
    gpu_bytes = torch.cuda.max_memory_allocated() if _cuda() else 0
    with open(_cache_path(cls_name, params), "w") as f:
        json.dump(
            {"host_delta_bytes": host_delta_bytes, "gpu_bytes": gpu_bytes}, f
        )


def _read_cached(cls_name: str, params: Tuple, key: str) -> int:
    try:
        with open(_cache_path(cls_name, params)) as f:
            return int(json.load(f).get(key, 0))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


__all__ = [
    "BackendsBenchmark",
    "RegularizersBenchmark",
    "ScaffoldBenchmark",
    "SecAggBenchmark",
]


def _resolve_n_clients_axis() -> List[int]:
    """Read DECLEARN_BENCH_N_CLIENTS or fall back to the [5] default.

    The env var is set by `bench_config.py` from the active profile.
    Silent fallback (rather than raising) so that direct `asv run`
    invocations outside `run_benchmarks.sh` still work.
    """
    raw = os.environ.get("DECLEARN_BENCH_N_CLIENTS", "").strip()
    if not raw:
        return [5]
    try:
        axis = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            f"DECLEARN_BENCH_N_CLIENTS must be comma-separated ints; "
            f"got {raw!r}"
        ) from exc
    if not axis or any(v <= 0 for v in axis):
        raise ValueError(
            f"DECLEARN_BENCH_N_CLIENTS must contain positive ints; "
            f"got {raw!r}"
        )
    return axis


# Single source of truth for the n_clients sweep across every category.
# Defaults to [5] to keep the per-version runtime bounded; override via
# the DECLEARN_BENCH_N_CLIENTS env var (set by run_benchmarks.sh from
# the active bench.yaml profile).
N_CLIENTS_AXIS: List[int] = _resolve_n_clients_axis()


class BackendsBenchmark:
    """Sweep model backends and client count on the FedAvg baseline."""

    timeout = 900.0
    params = (N_CLIENTS_AXIS, ["torch", "tensorflow"])
    param_names = ["n_clients", "backend"]

    def setup(self, n_clients: int, backend: str) -> None:
        ensure_data_for_n_clients(n_clients, BACKEND_LAYOUT[backend])

    def time_run(self, n_clients: int, backend: str) -> None:
        baseline = _mem_capture_start()
        spec = build_benchmark(backend=backend, n_clients=n_clients)
        run_benchmark(spec)
        _mem_capture_end("BackendsBenchmark", (n_clients, backend), baseline)

    def track_peakmem_run(self, n_clients: int, backend: str) -> int:
        return _read_cached(
            "BackendsBenchmark", (n_clients, backend), "host_delta_bytes"
        )
    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int, backend: str) -> float:
        # Only torch tensors are tracked by torch.cuda. TF would need its
        # own probe; return NaN so ASV renders the cell as n/a rather than
        # failed.
        if backend != "torch":
            return float("nan")
        return _read_cached(
            "BackendsBenchmark", (n_clients, backend), "gpu_bytes"
        )
    track_peakgpu_run.unit = "bytes"  # type: ignore[attr-defined]


class RegularizersBenchmark:
    """Sweep client-side loss regularizers and client count (torch FedAvg)."""

    timeout = 600.0
    params = (N_CLIENTS_AXIS, ["ridge", "fedprox"])
    param_names = ["n_clients", "regularizer"]

    def setup(self, n_clients: int, regularizer: str) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int, regularizer: str) -> None:
        baseline = _mem_capture_start()
        spec = build_benchmark(
            backend="torch", regularizer=regularizer, n_clients=n_clients
        )
        run_benchmark(spec)
        _mem_capture_end(
            "RegularizersBenchmark", (n_clients, regularizer), baseline
        )

    def track_peakmem_run(self, n_clients: int, regularizer: str) -> int:
        return _read_cached(
            "RegularizersBenchmark",
            (n_clients, regularizer),
            "host_delta_bytes",
        )
    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int, regularizer: str) -> int:
        return _read_cached(
            "RegularizersBenchmark", (n_clients, regularizer), "gpu_bytes"
        )
    track_peakgpu_run.unit = "bytes"  # type: ignore[attr-defined]


class ScaffoldBenchmark:
    """Sweep client count for SCAFFOLD on torch."""

    timeout = 600.0
    params = N_CLIENTS_AXIS
    param_names = ["n_clients"]

    def setup(self, n_clients: int) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int) -> None:
        baseline = _mem_capture_start()
        spec = build_benchmark(
            backend="torch", scaffold=True, n_clients=n_clients
        )
        run_benchmark(spec)
        _mem_capture_end("ScaffoldBenchmark", (n_clients,), baseline)

    def track_peakmem_run(self, n_clients: int) -> int:
        return _read_cached(
            "ScaffoldBenchmark", (n_clients,), "host_delta_bytes"
        )
    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int) -> int:
        return _read_cached("ScaffoldBenchmark", (n_clients,), "gpu_bytes")
    track_peakgpu_run.unit = "bytes"  # type: ignore[attr-defined]


class SecAggBenchmark:
    """SecAgg masking sweep over client count on torch."""

    timeout = 1200.0
    params = N_CLIENTS_AXIS
    param_names = ["n_clients"]

    def setup(self, n_clients: int) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int) -> None:
        baseline = _mem_capture_start()
        spec = build_benchmark(
            backend="torch", secagg="masking", n_clients=n_clients
        )
        run_benchmark(spec)
        _mem_capture_end("SecAggBenchmark", (n_clients,), baseline)

    def track_peakmem_run(self, n_clients: int) -> int:
        return _read_cached(
            "SecAggBenchmark", (n_clients,), "host_delta_bytes"
        )
    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int) -> int:
        return _read_cached("SecAggBenchmark", (n_clients,), "gpu_bytes")
    track_peakgpu_run.unit = "bytes"  # type: ignore[attr-defined]
