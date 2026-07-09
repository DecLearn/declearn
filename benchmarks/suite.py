"""ASV benchmark classes for the declearn suite.

Each class is a thin wrapper over `build_benchmark(...)` +
`run_benchmark(...)`; ASV discovers them automatically by walking this
module. The heavy lifting (parameter interpretation, data preparation,
network/model setup) lives in `benchmarks.workload`; the per-cell memory
probing lives in `benchmarks._memory`. The classes here only declare
which slice of the parameter space each benchmark exercises.

Benchmark classes:

* [BackendsBenchmark][]:
    Model backend (torch, tensorflow) x client count, on FedAvg.
* [RegularizersBenchmark][]:
    Client-side loss regularizer (ridge, fedprox) x client count (torch).
* [ScaffoldBenchmark][]:
    SCAFFOLD auxiliary-variable exchange over client count (torch).
* [SecAggBenchmark][]:
    Secure aggregation via masking over client count (torch).

The `n_clients` axis is a single shared point (`[5]` by default) to keep
the per-version sweep bounded. It is sourced from the
`DECLEARN_BENCH_N_CLIENTS` env var (comma-separated positive ints, e.g.
"5,20") when set, so that `bench.yaml` profiles can widen or narrow the
sweep without editing this file.
"""

import os
from typing import List

from benchmarks._memory import capture_end, capture_start, read_cached
from benchmarks.workload import build_benchmark, run_benchmark
from benchmarks.workload.build import BACKEND_LAYOUT
from benchmarks.workload.data import ensure_data_for_n_clients

__all__ = [
    "BackendsBenchmark",
    "RegularizersBenchmark",
    "ScaffoldBenchmark",
    "SecAggBenchmark",
]


def _resolve_n_clients_axis() -> List[int]:
    """Read DECLEARN_BENCH_N_CLIENTS or fall back to the [5] default.

    The env var is set by `bench_config.py` from the active profile.
    Silent fallback (rather than raising) on an unset var so that direct
    `asv run` invocations outside `run_benchmarks.sh` still work.
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
            f"DECLEARN_BENCH_N_CLIENTS must contain positive ints; got {raw!r}"
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
        baseline = capture_start()
        spec = build_benchmark(backend=backend, n_clients=n_clients)
        run_benchmark(spec)
        capture_end("BackendsBenchmark", (n_clients, backend), baseline)

    def track_peakmem_run(self, n_clients: int, backend: str) -> int:
        return read_cached(
            "BackendsBenchmark", (n_clients, backend), "host_delta_bytes"
        )

    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int, backend: str) -> float:
        # Only torch tensors are tracked by torch.cuda. TF would need its
        # own probe; return NaN so ASV renders the cell as n/a rather than
        # failed.
        if backend != "torch":
            return float("nan")
        return read_cached(
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
        baseline = capture_start()
        spec = build_benchmark(
            backend="torch", regularizer=regularizer, n_clients=n_clients
        )
        run_benchmark(spec)
        capture_end(
            "RegularizersBenchmark", (n_clients, regularizer), baseline
        )

    def track_peakmem_run(self, n_clients: int, regularizer: str) -> int:
        return read_cached(
            "RegularizersBenchmark",
            (n_clients, regularizer),
            "host_delta_bytes",
        )

    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int, regularizer: str) -> int:
        return read_cached(
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
        baseline = capture_start()
        spec = build_benchmark(
            backend="torch", scaffold=True, n_clients=n_clients
        )
        run_benchmark(spec)
        capture_end("ScaffoldBenchmark", (n_clients,), baseline)

    def track_peakmem_run(self, n_clients: int) -> int:
        return read_cached(
            "ScaffoldBenchmark", (n_clients,), "host_delta_bytes"
        )

    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int) -> int:
        return read_cached("ScaffoldBenchmark", (n_clients,), "gpu_bytes")

    track_peakgpu_run.unit = "bytes"  # type: ignore[attr-defined]


class SecAggBenchmark:
    """SecAgg masking sweep over client count on torch."""

    timeout = 1200.0
    params = N_CLIENTS_AXIS
    param_names = ["n_clients"]

    def setup(self, n_clients: int) -> None:
        ensure_data_for_n_clients(n_clients, "chw")

    def time_run(self, n_clients: int) -> None:
        baseline = capture_start()
        spec = build_benchmark(
            backend="torch", secagg="masking", n_clients=n_clients
        )
        run_benchmark(spec)
        capture_end("SecAggBenchmark", (n_clients,), baseline)

    def track_peakmem_run(self, n_clients: int) -> int:
        return read_cached("SecAggBenchmark", (n_clients,), "host_delta_bytes")

    track_peakmem_run.unit = "bytes"  # type: ignore[attr-defined]

    def track_peakgpu_run(self, n_clients: int) -> int:
        return read_cached("SecAggBenchmark", (n_clients,), "gpu_bytes")

    track_peakgpu_run.unit = "bytes"  # type: ignore[attr-defined]
