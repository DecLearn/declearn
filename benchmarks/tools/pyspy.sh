#!/bin/bash
# Wrap one benchmark configuration in py-spy and emit a speedscope JSON.
#
# Targets `benchmarks.tools.profile_entry`, which is a single-shot wrapper
# around build_benchmark + run_benchmark — exactly one cell's worth of
# work, no ASV harness. py-spy samples the Python stack while it runs and
# emits a speedscope-format profile, viewable at https://www.speedscope.app/
# (drag-and-drop the .json file) or in any local speedscope build.
#
# Usage (from anywhere — the script resolves its own paths):
#   ./benchmarks/tools/pyspy.sh --backend torch --n-clients 5
#   ./benchmarks/tools/pyspy.sh --secagg masking --backend torch --n-clients 5
#   OUTPUT=foo.json ./benchmarks/tools/pyspy.sh --backend tensorflow
#
# Env vars:
#   OUTPUT       target JSON path (default: <benchmarks>/profiles/<timestamp>.json).
#                Relative paths are resolved against the caller's cwd, not the
#                declearn repo root.
#   RATE         py-spy sampling rate in Hz (default: 100)
#   BENCH_VENV   default: $HOME/.venvs/declearn-bench-gpu
#                Only used if no virtualenv is already active.

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$TOOLS_DIR")"
PROJECT_ROOT="$(dirname "$BENCH_DIR")"

# Capture the caller's cwd before `cd`-ing away, so that a relative
# `OUTPUT=foo.json` lands where the user invoked the script from, not
# at the declearn repo root.
CALLER_PWD="$PWD"

# `python -m benchmarks.tools.profile_entry` resolves only from the
# parent of the `benchmarks/` package (the declearn repo root).
cd "$PROJECT_ROOT"

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

if ! command -v py-spy >/dev/null 2>&1; then
    echo "ERROR: py-spy not on PATH. Install with: pip install py-spy" >&2
    exit 1
fi

DEFAULT_OUTPUT="${BENCH_DIR}/profiles/$(date +%Y%m%d-%H%M%S).json"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
# Resolve a relative OUTPUT against the caller's cwd; absolute paths pass through.
case "$OUTPUT" in
    /*) ;;
    *) OUTPUT="$CALLER_PWD/$OUTPUT" ;;
esac
RATE="${RATE:-100}"
mkdir -p "$(dirname "$OUTPUT")"

# Force-loud GPU: silent CPU fallback would make timings non-comparable.
export DECLEARN_BENCH_FORCE_GPU="${DECLEARN_BENCH_FORCE_GPU:-1}"

echo "=== py-spy recording to ${OUTPUT} (speedscope, rate=${RATE}Hz) ==="
exec py-spy record -o "$OUTPUT" --rate "$RATE" --format speedscope -- \
    python -m benchmarks.tools.profile_entry "$@"
