# Benchmarking

Declearn's source repository includes a performance benchmark suite,
implemented under the `benchmarks/` folder and built on the third-party
[ASV (airspeed velocity)](https://asv.readthedocs.io/) tool. It measures
the timing and memory footprint of the federated-learning pipeline
(client registration, training rounds, model aggregation, evaluation and
optional secure aggregation) end-to-end, across declearn versions and on
a fixed MNIST workload.

The suite is meant to serve two distinct purposes:

- As a release-tag regression check, a fixed subset of benchmarks is run
  on every release-tag push, comparing the new release against the previous one
  and failing the pipeline if a regression beyond a 1.5x factor is
  detected.
- As an on-demand investigation tool, developers may run any subset of benchmarks against any commits, browse the
  results as an ASV timeline, and use the py-spy (CPU/time) and memray
  (memory) integrations to generate flame graphs for a specific
  configuration when narrowing down where a regression lives.

## Benchmark classes

The benchmark classes are defined in `benchmarks/__init__.py`; these are
the entities that ASV discovers, parameterises and runs. Four classes
are currently in place, each exercising a different declearn feature
against the same MNIST workload.

| Class | Sweep axes | What it exercises |
|---|---|---|
| `BackendsBenchmark` | `n_clients` × `backend` (torch, tensorflow) | The FL pipeline on both ML backends |
| `RegularizersBenchmark` | `n_clients` × `regularizer` (ridge, fedprox) | Client-side loss regularizers (torch) |
| `ScaffoldBenchmark` | `n_clients` | SCAFFOLD auxiliary-variable exchange (torch) |
| `SecAggBenchmark` | `n_clients` | Secure aggregation via masking (torch) |

Each class measures three quantities for every parameter combination:

- `time_run`, the wall-clock time of one full FL run (registration,
  training rounds, aggregation and evaluation);
- `track_peakmem_run`, the peak host RSS growth during the run;
- `track_peakgpu_run`, the peak GPU bytes allocated during the run
  (this is torch-only, as the tensorflow cell is intentionally skipped).

The `n_clients` axis is shared across all four classes. It is sourced
from the `DECLEARN_BENCH_N_CLIENTS` environment variable that the runner
emits from the active preset, so that widening the axis in `bench.yaml`
propagates everywhere automatically.

## Presets and configuration

Benchmark runs are parameterised through a single YAML file
(`benchmarks/bench.yaml`) together with a small helper script
(`benchmarks/bench_config.py`) that the runner consults in order to turn
a named preset into the concrete options ASV expects.

A preset is a named bundle of three settings: `classes` controls which
benchmark classes ASV will run, `n_clients_axis` defines the `n_clients`
sweep axis (for example `[5]` or `[5, 20]`), and `asv_args` lists extra
flags to be passed to ASV (such as `--quick` or `--show-stderr`). The
`default_preset:` entry at the top of the YAML names the preset that is
picked when the `PRESET` environment variable is not set.

The presets currently available are the following:

| Preset | Classes | n_clients | asv_args | Use |
|---|---|---|---|---|
| `ci` (default) | all four | `[5]` | `--quick`, `--show-stderr` | The release-tag regression check that the CI runs; all classes with single-sample timings |
| `full` | all four | `[5]` | `--show-stderr` | The definitive on-demand sweep |
| `scale` | all four | `[5, 20]` | `--show-stderr` | Measuring performance at scale  |

Note that widening `n_clients_axis` to include larger client counts makes
some benchmark classes such as `SecAggBenchmark` significantly slower.

For ad-hoc runs that do not justify defining a new preset, the `CLASSES`
environment variable may be used to override the active preset's list of
classes while preserving its `n_clients_axis` and `asv_args`.

## The runner

The `benchmarks/run_benchmarks.sh` script is the single entry point for
every benchmark invocation, both locally and inside the
CI. It acts as a thin wrapper that resolves the active preset (via
`bench_config.py`), activates the bench virtual environment when needed,
sets the GPU-policy environment, and hands control over to ASV with the
resolved filters and flags already in place.

## The workload

The `benchmarks/workload/` subpackage holds the "what runs inside a
benchmark cell" half of the suite. Where `__init__.py` declares the ASV
cells and the metrics taken around them, `workload/` produces a runnable
FL experiment from a set of toggles and runs it.

Each ASV cell's `time_run` essentially amounts to the following two
calls:

```python
spec = build_benchmark(backend=..., n_clients=..., ...)
run_benchmark(spec)
```

These same two calls also power the profiling entry point
(see the [dedicated section](#investigating-regressions-with-py-spy-and-memray)
below), so that the actual FL workload is defined in a single place and
shared by both the sweep harness and the single-shot profiling target.

The subpackage is organised as follows:

| File | Role |
|---|---|
| `baseline.py` | Fixed constants every run inherits (learning rates, batch size, rounds, dataset fraction, network port, and so on). This is the single place to change "the baseline". |
| `spec.py` | The `BenchmarkSpec` and `ClientSpec` dataclasses, describing the data shape in which every run is expressed. |
| `models/torch_cnn.py`, `models/tensorflow_cnn.py` | The small MNIST CNNs used by each backend. |
| `data.py` | Idempotent MNIST split preparation, producing per-client NumPy shards on disk under `data/`. |
| `build.py` | `build_benchmark(...)`: turns toggle arguments (backend, n_clients, regularizer, scaffold, secagg, and so on) into a fully populated `BenchmarkSpec`. It validates combinations, builds the model, picks the FL optimisation config and generates Ed25519 keys for SecAgg. |
| `runner.py` | `run_benchmark(spec)`: takes a spec, spawns the FL server and N clients on a single asyncio event loop, and awaits completion. |

## Investigating regressions with py-spy and memray

When a regression is detected, the next question is usually *where in
the code it happens*. The `benchmarks/tools/` subpackage provides a small
flow for that step: one picks the configuration that regressed, records a
single run, then inspects the resulting flame graph. Two complementary
profilers are wired in: [py-spy](https://github.com/benfred/py-spy) for
CPU/time and [memray](https://github.com/bloomberg/memray) for memory,
both sharing the same entry point.

| File | Role |
|---|---|
| `tools/profile_entry.py` | An argparse-driven target that runs a single concrete configuration once, calling `build_benchmark` and `run_benchmark` directly and thereby bypassing ASV's sweep harness. Profiler-agnostic: both wrappers below target it. |
| `tools/pyspy.sh` | Wraps `profile_entry.py` with `py-spy record`, emitting a [speedscope](https://www.speedscope.app/) JSON file under `profiles/` (which is gitignored). Use it to find where time is spent. |
| `tools/memray.sh` | Wraps `profile_entry.py` with `memray run`, writing a `.bin` capture under `profiles/` and then rendering an HTML memory flame graph alongside it. Use it to find where memory is allocated. Set `NATIVE=1` to also capture native (C/C++) allocation stacks, e.g. torch / tensorflow tensors. |

For py-spy, the resulting `.json` file may be dragged into
<https://www.speedscope.app/> to obtain an interactive flame graph. For
memray, the generated `.html` file is opened directly in a browser.

## Continuous integration

The bench job is defined in declearn's existing `.gitlab-ci.yml` file,
as a sibling of the `test-minimal` and `test-maximal` jobs, within its
own `bench` stage. It does not run on day-to-day MR or develop pushes;
it is triggered only on release-tag pushes and on manual "Run pipeline"
runs from the GitLab UI that set the `BENCH_BOOTSTRAP=true` variable.

The job operates in one of two modes. In regression-check mode, run on every
release-tag push, it compares the new release against the previous one
and fails the pipeline on a regression. In bootstrap mode, triggered by
running a pipeline from the GitLab UI with the `BENCH_BOOTSTRAP=true`
variable set, it seeds the initial benchmark history across the list of
releases given in `BOOTSTRAP_TAGS`; this is meant to be run once at
setup, and again whenever the history needs widening. A plain "Run
pipeline" without `BENCH_BOOTSTRAP` does not start the bench job.

Benchmark history is kept as a cumulative artifact rather than committed
to the repository. The bench job uploads `.asv/results/` and
`.asv/html/` as a single artifact (configured with `expire_in: never`
and `when: always`), which is the source of truth for benchmark history.
The regression check fetches the previous tag's artifact through GitLab's
cross-pipeline API, falling back to develop's bootstrap artifact when it
is missing, and soft-seeds (without comparison) when no baseline exists
at all, so that the first run does not fail for lack of a reference.
The `when: always` setting keeps the result JSONs and rendered HTML
accessible from the pipeline page even when a regression has failed the
job, so that investigation can start directly from the artifacts. The
next section walks that investigation end to end.

## Investigating a detected regression

This is the full flow, from a red release pipeline to the commit that
caused the slowdown. It crosses two machines: the browser-dependent
steps (artifact download, dashboard preview, flame graph viewing) run
locally, and the GPU cluster handles the profiling. The
flame-graph JSONs are small (typically a few MB), so shipping them
between the two over `scp` is cheap.

### 1. Confirm the regression in the CI log

Open the failed pipeline from the project's Pipelines page (filter by
the release tag). In the `benchmarks` job log, confirm:

- the comparison line, `=== regression check: <prev_tag> vs <new_tag> (--factor 1.5) ===`;
- the `asv continuous` comparison table, showing the regressed
  benchmark(s) with a ratio above the factor;
- that `asv publish` still ran afterwards (the loading-results /
  generating-graphs lines) and the job then exited non-zero.

Note the **regressed cell** (e.g. `BackendsBenchmark.time_run(5, 'torch')`)
and the **two refs** being compared (e.g. `v2.7.0` vs `v2.8.0`).

### 2. Download the artifact

Download the failed-job artifact from the pipeline page. Previewing it
needs `asv` installed:

```bash
unzip -q artifacts.zip -d bench-artifact
cd bench-artifact/benchmarks
asv preview                          # serves http://localhost:8080
```

Browse to the regressed benchmark's timeline and confirm where it jumps.

### 3. Set up the cluster environment (one-time)

The profiling steps run on the cluster, where a CUDA-capable GPU and a
consistent runtime are available. A working GPU and Python 3.11 are
required: the suite does run end-to-end on CPU, but the timings will not
be comparable to the CI's, and `DECLEARN_BENCH_FORCE_GPU=1` (the
default) refuses to proceed in that case.

The host venv only needs to drive ASV and the profilers; declearn +
torch/tensorflow are installed per-checkout in step 4 (and, for sweeps,
per-commit by ASV; see `asv.conf.json`). Create it with uv, activate it,
then run the bootstrap:

```bash
# The venv must be named declearn-bench-gpu: bootstrap_cluster.sh and the
# steps below default to ~/.venvs/declearn-bench-gpu. --seed ships pip
# inside the venv (without it the installs fall through to a system Python).
uv venv --python 3.11 --seed ~/.venvs/declearn-bench-gpu
source ~/.venvs/declearn-bench-gpu/bin/activate
./benchmarks/bootstrap_cluster.sh    # installs asv, pyyaml, py-spy, memray
```

### 4. Cluster: profile both refs

Profile the **baseline** and the **slow** ref: the baseline run is what
gives the slow run something to be compared against, so both need
profiling, not just the slow one.

First the baseline:

```bash
git clone https://gitlab.inria.fr/magnet/declearn/declearn.git
cd ~/declearn
git checkout <prev_ref>            # the baseline, e.g. v2.7.0
pip install "torch<=2.11" --index-url https://download.pytorch.org/whl/cu126
pip install -e '.[torch,tensorflow,websockets]'
pip install 'tensorflow[and-cuda]'
# tensorflow[and-cuda] ships ptxas inside site-packages but not on PATH;
# TF's XLA JIT needs it at runtime, so link it into the venv's bin.
ln -sf "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin/ptxas" "$VIRTUAL_ENV/bin/"
cd benchmarks
OUTPUT=profiles/baseline-torch.json ./tools/pyspy.sh --backend torch --n-clients 5
```

The editable (`-e`) install links the live source tree, so the next
`git checkout` updates the installed declearn automatically. Reinstall
only if the dependencies in `pyproject.toml` actually change between the
two refs.

Then the slow ref:

```bash
cd ~/declearn
git checkout <new_ref>             # the slow ref, e.g. v2.8.0
cd benchmarks
OUTPUT=profiles/slow-torch.json ./tools/pyspy.sh --backend torch --n-clients 5
```

### 5. Locally: diff the flame graphs in speedscope

```bash
scp cluster:~/declearn/benchmarks/profiles/baseline-torch.json \
    cluster:~/declearn/benchmarks/profiles/slow-torch.json \
    /tmp/
```

Open <https://www.speedscope.app/>, drop both files (one per tab),
switch each to "Left Heavy" view, and look for a new wide bar in
`slow-torch.json` that is missing or thin in `baseline-torch.json`.

### 6. Pin the suspect commit

Once a hotspot is localized, narrow the regression to the commits that
touched that hot path:

```bash
git log <prev_ref>..<new_ref> -- <path/to/hot/file.py>
```

## Other use cases

Beyond the regression investigation, the suite supports on-demand sweeps and
profiling on the cluster. These all assume the host venv from
[step 3](#3-set-up-the-cluster-environment-one-time) above is set up and
activated; profiling or sweeping a specific checkout additionally needs
the editable declearn install from
[step 4](#4-cluster-profile-both-refs). Every command below assumes the
`benchmarks/` folder is the current working directory.

All three runner entry points (`run_benchmarks.sh`, `tools/pyspy.sh` and
`tools/memray.sh`) default to `DECLEARN_BENCH_FORCE_GPU=1`, which aborts
the run if no CUDA GPU is visible. This is deliberate: a misconfigured
CUDA stack would otherwise silently fall back to CPU and produce timings
that look like a 10x regression, so the suite fails loudly instead. To
run on a CPU-only machine anyway (for example a quick smoke test of the
harness itself), set `DECLEARN_BENCH_FORCE_GPU=0`; the run then proceeds
on CPU, but its timings are not comparable to the CI's:

```bash
DECLEARN_BENCH_FORCE_GPU=0 ./run_benchmarks.sh
```

### Where results are written

ASV benchmark results accumulate under `.asv/results/`, as one JSON file
per `(commit, env)` pair. After a run, `asv publish` renders them into
`.asv/html/`, a self-contained dashboard with per-benchmark timelines
across every commit for which the local machine holds data. The
`asv preview` command then serves that dashboard on
`http://localhost:8080` for browsing.

The profiler outputs are written under `profiles/`: py-spy speedscope
JSON files, viewed by drag-and-dropping them into
<https://www.speedscope.app/>, and memray `.bin` captures alongside their
rendered `.html` memory flame graphs, opened directly in a browser.

### Run a sweep on the current checkout

```bash
# 1. Run the default preset (ci) on the current HEAD.
#    Writes per-cell timing and memory JSONs to .asv/results/.
./run_benchmarks.sh

# 2. Render the accumulated results into a static HTML dashboard.
asv publish

# 3. Serve the dashboard on http://localhost:8080.
asv preview
```

### Profile a single configuration

```bash
# CPU/time: writes profiles/<timestamp>.json; load it into speedscope.
./tools/pyspy.sh --backend torch --n-clients 5

# Memory: writes profiles/<timestamp>.bin and renders the .html alongside
# it; open the .html in a browser. NATIVE=1 also captures torch /
# tensorflow native (C/C++) allocations.
./tools/memray.sh --backend torch --n-clients 5
```

### Other run recipes

```bash
# Use a different preset
PRESET=full ./run_benchmarks.sh

# Compare two releases in one sweep; fails on >1.5x regression
PRESET=full ./run_benchmarks.sh continuous v2.7.0 v2.8.0 --factor 1.5

# Run across several tags to seed history (asv skips refs already done)
PRESET=ci ./run_benchmarks.sh run v2.5.0 v2.6.0 v2.7.0 v2.8.0 --skip-existing-commits

# Ad-hoc class subset, reusing the active preset's flags and axis
CLASSES=ScaffoldBenchmark PRESET=full ./run_benchmarks.sh

# Pass workload toggles to pyspy.sh / memray.sh (here, masking SecAgg)
./tools/pyspy.sh --backend torch --n-clients 5 --secagg
./tools/memray.sh --backend torch --n-clients 5 --secagg
```

### Profiling configurations outside the build_benchmark surface

The `benchmarks/tools/pyspy.sh` and `benchmarks/tools/memray.sh` scripts
accept exactly the toggles that `build_benchmark` exposes. For an
investigation that needs something broader (a different model, an
aggregator other than averaging, differential privacy, gRPC instead of
websockets, or a custom optimizer module), the investigator writes their
own driver script and wraps it with the profiler directly. There are
three layers available, ordered from the cheapest to the most general:

| Layer | Use when | Driver builds the run via |
|---|---|---|
| `tools/pyspy.sh` / `tools/memray.sh` | The configuration is reachable through `build_benchmark`'s existing toggles. | The wrapper, plus `profile_entry.py`. |
| A custom driver on top of `benchmarks.workload` | The configuration uses `build_benchmark` plus extra setup the wrapper does not expose. | `from benchmarks.workload import build_benchmark, run_benchmark`, adding whatever instrumentation or knobs are needed. |
| A custom driver on top of the raw declearn APIs | The configuration is outside `build_benchmark`'s surface entirely. | `from declearn.main import FederatedServer, FederatedClient` (and the rest of declearn's API), constructing the experiment from scratch. |

In every case, the profile is recorded by wrapping the driver with the
chosen profiler: `py-spy record -o out.json --format speedscope -- python
driver.py` for CPU/time, or `memray run -o out.bin -- python driver.py`
(then `memray flamegraph out.bin`) for memory.

## Extending the suite

### Adding a benchmark class

Adding a new benchmark class requires touching three files:

1. In `__init__.py`, define the class itself. Declare its `params`,
   `param_names` and `timeout`, and implement the `time_run(...)`,
   `setup(...)` and `track_*` methods, following the shape of the four
   existing classes. The body should delegate to `build_benchmark(...)`
   and `run_benchmark(...)`, so that the workload stays defined in a
   single place.
2. In `bench_config.py`, add the new class name to the `KNOWN_CLASSES`
   frozenset. Without this, the name fails the allowlist check at
   preset-resolution time, even though ASV would discover the class
   without issue; the explicit allowlist is what catches typos in
   `bench.yaml`.
3. Optionally, in `bench.yaml`, list the new class in an existing
   preset's `classes:` block, or add a dedicated preset for it.

If the new class needs a parameter that `build_benchmark` does not
currently accept (a new aggregator, a new SecAgg method, a new optimizer
module), extend `workload/build.py` first to add the toggle, then
reference it from the new class.

### Adding or editing a preset

Presets are edited directly in `bench.yaml`. A preset is a YAML mapping
with `classes:`, `n_clients_axis:` and `asv_args:` entries. The runner
picks it up the next time `PRESET=<name>` is set, and no other file
needs to change.
