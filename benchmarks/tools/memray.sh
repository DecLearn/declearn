#!/bin/bash
# Wrap one benchmark configuration in memray and emit a memory flame graph.
#
# Targets `benchmarks.tools.profile_entry`, the same single-shot wrapper
# used by pyspy.sh — exactly one cell's worth of work, no ASV harness.
# memray traces every allocation while it runs and writes a binary capture
# (.bin); this script then renders an interactive HTML memory flame graph
# from that capture, viewable directly in any browser.
#
# Usage (from anywhere — the script resolves its own paths):
#   ./benchmarks/tools/memray.sh --backend torch --n-clients 5
#   ./benchmarks/tools/memray.sh --secagg --backend torch --n-clients 5
#   NATIVE=1 ./benchmarks/tools/memray.sh --backend torch   # capture C/C++ stacks
#   OUTPUT=foo.bin ./benchmarks/tools/memray.sh --backend tensorflow
#
# Env vars:
#   OUTPUT       target .bin capture path (default: <benchmarks>/profiles/<timestamp>.bin).
#                Relative paths are resolved against the caller's cwd, not the
#                declearn repo root. The HTML flame graph is written alongside
#                the capture, as <OUTPUT>.html.
#   NATIVE       set to 1 to also record native (C/C++) allocation stacks, e.g.
#                torch / tensorflow tensor allocations. Slower, larger capture.
#   BENCH_VENV   default: $HOME/.venvs/declearn-bench-gpu
#                Only used if no virtualenv is already active.

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$TOOLS_DIR")"
PROJECT_ROOT="$(dirname "$BENCH_DIR")"

# Capture the caller's cwd before `cd`-ing away, so that a relative
# `OUTPUT=foo.bin` lands where the user invoked the script from, not
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

if ! command -v memray >/dev/null 2>&1; then
    echo "ERROR: memray not on PATH. Install with: pip install memray" >&2
    exit 1
fi

DEFAULT_OUTPUT="${BENCH_DIR}/profiles/$(date +%Y%m%d-%H%M%S).bin"
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
# Resolve a relative OUTPUT against the caller's cwd; absolute paths pass through.
case "$OUTPUT" in
    /*) ;;
    *) OUTPUT="$CALLER_PWD/$OUTPUT" ;;
esac
FLAMEGRAPH="${OUTPUT%.bin}.html"
mkdir -p "$(dirname "$OUTPUT")"

# Force-loud GPU: silent CPU fallback would make allocations non-comparable.
export DECLEARN_BENCH_FORCE_GPU="${DECLEARN_BENCH_FORCE_GPU:-1}"

# Native (C/C++) allocation stacks are opt-in: they surface framework-level
# allocations (torch / tensorflow tensors) at the cost of speed and capture size.
NATIVE_FLAG=""
if [ "${NATIVE:-0}" = "1" ]; then
    NATIVE_FLAG="--native"
fi

echo "=== memray recording to ${OUTPUT}${NATIVE_FLAG:+ (native)} ==="
# shellcheck disable=SC2086  # NATIVE_FLAG is intentionally word-split (empty or --native)
memray run $NATIVE_FLAG -o "$OUTPUT" --force \
    -m benchmarks.tools.profile_entry "$@"

echo "=== rendering memory flame graph to ${FLAMEGRAPH} ==="
memray flamegraph -o "$FLAMEGRAPH" --force "$OUTPUT"

echo "Done. Open ${FLAMEGRAPH} in a browser for the memory flame graph."
