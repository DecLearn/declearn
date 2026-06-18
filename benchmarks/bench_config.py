r"""Resolve a bench.yaml preset into shell-eval'able run parameters.

The benchmark suite is invoked through `run_benchmarks.sh`. That script
does environment activation and per-version pip pinning; this helper
translates a named preset from `bench.yaml` into the concrete env vars
and ASV argv that the sh script then applies. Output is meant to be
consumed by `eval "$(python bench_config.py --preset <name>)"`.

The emitted block defines:

- `BENCH_PRESET` (string)  - echoed back so the sh script can log it.
- `DECLEARN_BENCH_N_CLIENTS` (exported)  - comma-joined axis read by
  `benchmarks/__init__.py` at ASV discovery time.
- `ASV_BENCH_FILTERS` (bash array)  - one ``-b ^ClassName\.`` pair per
  class in the preset. Anchoring with ``^`` and ``\.`` avoids accidental
  substring matches across class names.
- `ASV_EXTRA_ARGS` (bash array)  - the preset's `asv_args` verbatim.

Class names are validated against `KNOWN_CLASSES` to catch typos at
config-resolution time rather than silently producing an ASV run that
matches zero benchmarks.

`--classes` (forwarded from the shell script's `CLASSES` env var) lets
an ad-hoc invocation override the preset's class list while keeping
the preset's `n_clients_axis` and `asv_args` intact.
"""

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

__all__ = ["KNOWN_CLASSES", "load_preset", "render_shell"]


KNOWN_CLASSES = frozenset({
    "BackendsBenchmark",
    "RegularizersBenchmark",
    "ScaffoldBenchmark",
    "SecAggBenchmark",
})

DEFAULT_YAML_PATH = Path(__file__).resolve().parent / "bench.yaml"


def _die(msg: str) -> "None":
    print(f"bench_config: {msg}", file=sys.stderr)
    sys.exit(2)


def load_preset(
    yaml_path: Path,
    preset_name: str,
    classes_override: List[str] | None = None,
) -> Dict[str, Any]:
    """Read `yaml_path` and return the resolved preset dict.

    If `preset_name` is empty, falls back to `default_preset`.
    `classes_override` (from the `CLASSES` env var / `--classes` flag)
    replaces the preset's `classes` list when provided. The preset's
    `n_clients_axis` and `asv_args` are preserved either way. Raises
    via `_die` on missing files, missing presets, or malformed entries.
    """
    if not yaml_path.is_file():
        _die(f"yaml not found at {yaml_path}")
    with yaml_path.open() as handle:
        doc = yaml.safe_load(handle) or {}
    presets = doc.get("presets")
    if not isinstance(presets, dict) or not presets:
        _die(f"{yaml_path}: missing or empty 'presets' mapping")
    if not preset_name:
        preset_name = doc.get("default_preset", "")
        if not preset_name:
            _die(f"{yaml_path}: no preset requested and no 'default_preset'")
    if preset_name not in presets:
        available = ", ".join(sorted(presets))
        _die(
            f"unknown preset '{preset_name}'. Available: {available}"
        )
    preset = presets[preset_name]
    if not isinstance(preset, dict):
        _die(f"preset '{preset_name}' must be a mapping")

    if classes_override is not None:
        classes = classes_override
        source = "CLASSES override"
    else:
        classes = preset.get("classes") or []
        source = f"preset '{preset_name}'"
    if not isinstance(classes, list) or not classes:
        _die(f"{source}: 'classes' must be a non-empty list")
    unknown = [c for c in classes if c not in KNOWN_CLASSES]
    if unknown:
        _die(
            f"{source}: unknown class(es) {unknown}. "
            f"Known: {sorted(KNOWN_CLASSES)}"
        )

    axis = preset.get("n_clients_axis") or [5]
    if not isinstance(axis, list) or not all(
        isinstance(v, int) and v > 0 for v in axis
    ):
        _die(
            f"preset '{preset_name}': 'n_clients_axis' must be a list "
            "of positive ints"
        )

    asv_args = preset.get("asv_args") or []
    if not isinstance(asv_args, list) or not all(
        isinstance(v, str) for v in asv_args
    ):
        _die(
            f"preset '{preset_name}': 'asv_args' must be a list of strings"
        )

    return {
        "name": preset_name,
        "classes": list(classes),
        "n_clients_axis": list(axis),
        "asv_args": list(asv_args),
    }


def render_shell(preset: Dict[str, Any]) -> str:
    """Render `preset` as bash code that exports env + sets two arrays."""
    name = preset["name"]
    axis_csv = ",".join(str(v) for v in preset["n_clients_axis"])
    # Anchor each class name with ^...\. so a class whose name is a
    # prefix of another (none today, but cheap to defend) cannot leak.
    filters: List[str] = []
    for cls in preset["classes"]:
        filters.extend(["-b", f"^{cls}\\."])
    filter_array = " ".join(shlex.quote(tok) for tok in filters)
    extra_array = " ".join(shlex.quote(tok) for tok in preset["asv_args"])
    return (
        f"BENCH_PRESET={shlex.quote(name)}\n"
        f"export DECLEARN_BENCH_N_CLIENTS={shlex.quote(axis_csv)}\n"
        f"ASV_BENCH_FILTERS=({filter_array})\n"
        f"ASV_EXTRA_ARGS=({extra_array})\n"
    )


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a bench.yaml preset to shell-eval'able vars."
    )
    parser.add_argument(
        "--preset",
        default="",
        help="Preset name (defaults to bench.yaml's default_preset).",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=DEFAULT_YAML_PATH,
        help=f"Path to bench.yaml (default: {DEFAULT_YAML_PATH}).",
    )
    parser.add_argument(
        "--classes",
        default="",
        help=(
            "Comma-separated class names to run, overriding the "
            "preset's 'classes' list. The preset's n_clients_axis "
            "and asv_args are preserved."
        ),
    )
    args = parser.parse_args(argv)
    override: List[str] | None = None
    if args.classes.strip():
        override = [name.strip() for name in args.classes.split(",") if name.strip()]
    preset = load_preset(args.yaml, args.preset, override)
    sys.stdout.write(render_shell(preset))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
