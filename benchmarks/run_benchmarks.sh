#!/bin/bash
# Thin ASV wrapper driven by bench.yaml.
#
# Resolves the active preset via bench_config.py into:
#   - DECLEARN_BENCH_N_CLIENTS (env var read by __init__.py)
#   - -b <regex> filters per class
#   - preset-level asv flags (e.g. --quick, --show-stderr)
# …then execs `asv <subcommand> [args...] <filters> <flags>` so the caller
# decides whether to run `run`, `continuous`, `compare`, etc. Version
# checkout is handled natively by ASV (repo is the parent declearn repo),
# so no pip-install dance is needed.
#
# Usage:
#   ./run_benchmarks.sh                                       # asv run on current HEAD
#   ./run_benchmarks.sh continuous v2.7.0 v2.8.0 --factor 1.5
#   PRESET=full ./run_benchmarks.sh run ALL --skip-existing-commits
#   CLASSES=ScaffoldBenchmark PRESET=full ./run_benchmarks.sh continuous v2.7.0 HEAD
#
# Env vars:
#   PRESET        (default: bench.yaml's default_preset)
#   CLASSES       (optional CSV; overrides the preset's class list)
#   BENCH_VENV    (default: $HOME/.venvs/declearn-bench-gpu)
#                 Only used if no virtualenv is already active.

set -euo pipefail
cd "$(dirname "$0")"

PRESET="${PRESET:-}"
CLASSES="${CLASSES:-}"

# Activate the bench venv unless one is already active (e.g. in CI).
if [ -z "${VIRTUAL_ENV:-}" ]; then
    VENV="${BENCH_VENV:-$HOME/.venvs/declearn-bench-gpu}"
    if [ ! -d "$VENV" ]; then
        echo "ERROR: no venv at $VENV; run bootstrap_cluster.sh first." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

eval "$(python bench_config.py ${PRESET:+--preset "$PRESET"} ${CLASSES:+--classes "$CLASSES"})"
echo "=== bench preset: ${BENCH_PRESET} (n_clients=${DECLEARN_BENCH_N_CLIENTS})${CLASSES:+, classes overridden: ${CLASSES}} ==="

# Force-loud GPU: a silent CPU fallback would corrupt the comparison.
export DECLEARN_BENCH_FORCE_GPU="${DECLEARN_BENCH_FORCE_GPU:-1}"

# Ensure ASV has machine metadata. On a fresh runner (a clean CI image or a
# new cluster node) ~/.asv-machine.json does not exist yet, and in that case
# `asv run` / `asv continuous` abort non-interactively with "No information
# stored about machine ...". `asv machine --yes` writes platform defaults and
# is idempotent, so running it on every invocation is safe.
asv machine --yes >/dev/null

SUBCMD="${1:-run}"
shift || true

exec asv "$SUBCMD" "${ASV_BENCH_FILTERS[@]}" "${ASV_EXTRA_ARGS[@]}" "$@"
