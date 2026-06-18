#!/bin/bash
# Install the benchmark suite's host-venv deps.
#
# The host venv only needs to drive ASV — the heavy declearn install
# (torch / tensorflow / websockets) happens inside ASV's per-commit
# virtualenv (see asv.conf.json's `install_command` and `matrix`).
#
# Expects $BENCH_VENV to point at a Python 3.11 venv on a host with a
# working CUDA toolchain (driver + cuDNN libraries) on the standard
# search path. Does NOT manage Python or CUDA itself.
#
# Create the venv with uv so that it ships its own pip (the --seed flag
# is what guarantees this; without it the venv is pipless and the
# install calls below fall through to whatever system pip is on PATH):
#
#   uv venv --python 3.11 --seed "$BENCH_VENV"
#
# Usage:
#   ./bootstrap_cluster.sh
#   BENCH_VENV=/path/to/venv ./bootstrap_cluster.sh

set -euo pipefail

BENCH_VENV="${BENCH_VENV:-$HOME/.venvs/declearn-bench-gpu}"

if [ ! -d "$BENCH_VENV" ]; then
    echo "ERROR: no venv at $BENCH_VENV." >&2
    echo "  Expected to run inside declearn's CI image, or with" >&2
    echo "  BENCH_VENV pointing at a Python 3.11 venv you created, e.g.:" >&2
    echo "    uv venv --python 3.11 --seed $BENCH_VENV" >&2
    exit 1
fi

if [ ! -x "$BENCH_VENV/bin/python" ]; then
    echo "ERROR: $BENCH_VENV has no bin/python; not a usable venv." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$BENCH_VENV/bin/activate"

# Invoke pip via the venv's own interpreter rather than the bare `pip`
# on PATH. A pipless venv (uv venv without --seed) would otherwise let
# `pip` resolve to a system Python and install into the wrong place.
"$BENCH_VENV/bin/python" -m pip install --quiet --upgrade pip

# Host-venv deps: just enough to drive ASV and the profile loader.
# declearn + torch/tf/websockets + cryptography are installed by ASV
# into its per-commit venvs (see asv.conf.json).
echo "installing host-venv deps: asv, pyyaml, py-spy, memray"
"$BENCH_VENV/bin/python" -m pip install --quiet asv pyyaml py-spy memray

cat <<EOF

=== bootstrap finished ===
Next steps:
  source "$BENCH_VENV/bin/activate"
  export DECLEARN_BENCH_FORCE_GPU=1
  cd $(dirname "$0")
  ./run_benchmarks.sh                          # asv run on current HEAD
  ./run_benchmarks.sh continuous v2.7.0 HEAD   # diff two refs
EOF
