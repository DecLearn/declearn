"""Single-shot benchmark entry point for use under py-spy (or similar).

ASV runs benchmarks as a sweep across `params`. py-spy needs to wrap
one concrete Python invocation. This module accepts the same kwargs
that `build_benchmark(...)` takes and invokes `build_benchmark` +
`run_benchmark` exactly once — i.e. it executes one ASV "cell" worth
of work without ASV's discovery / forkserver / param-expansion harness.

Run from the parent of `benchmarks/` (i.e. the declearn repo root):

    python -m benchmarks.tools.profile_entry --backend torch --n-clients 5
    python -m benchmarks.tools.profile_entry --backend tensorflow --n-clients 5
    python -m benchmarks.tools.profile_entry --secagg --backend torch --n-clients 5

Typical use is via `tools/pyspy.sh` (CPU/time flame graph) or
`tools/memray.sh` (memory flame graph), which wrap this with the
respective profiler.
"""

import argparse
import logging

from benchmarks.workload import build_benchmark, run_benchmark


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one benchmark configuration once. Used as the profiler "
            "target (see tools/pyspy.sh and tools/memray.sh)."
        )
    )
    parser.add_argument(
        "--backend", default="torch", choices=["torch", "tensorflow"]
    )
    parser.add_argument("--n-clients", type=int, default=5)
    parser.add_argument(
        "--regularizer",
        default=None,
        choices=[None, "ridge", "fedprox"],
        help="Optional client-side loss regularizer.",
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Enable SCAFFOLD aux-var exchange (torch only).",
    )
    parser.add_argument(
        "--secagg",
        action="store_true",
        help="Enable masking-based secure aggregation.",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Imported lazily (not at module top) so ASV's benchmark discovery,
    # which imports every module under benchmark_dir, does not pull this
    # symbol in. It was added after v2.7.0, so a top-level import breaks
    # discovery when seeding older releases into the bench history.
    from declearn.utils import config_server_loggers

    # Surface the federated server's progress (registration, rounds,
    # aggregation, evaluation) on stderr while profiling, so the run is not
    # a silent black box. Scoped to this entry point on purpose: the ASV
    # cells share the same workload but stay quiet, to avoid polluting their
    # timings and flooding the CI logs.
    config_server_loggers(level=logging.INFO)
    spec = build_benchmark(
        backend=args.backend,
        n_clients=args.n_clients,
        regularizer=args.regularizer,
        scaffold=args.scaffold,
        secagg="masking" if args.secagg else None,
        rounds=args.rounds,
        batch_size=args.batch_size,
    )
    run_benchmark(spec)


if __name__ == "__main__":
    main()
