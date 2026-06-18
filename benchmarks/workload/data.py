"""Idempotent MNIST data preparation for benchmark runs.

A single canonical IID split is produced per `n_clients` value, then
re-shaped on demand into the layout each backend expects. All derived
layouts come from the same canonical split so that benchmark runs at
the same client count see identical sample assignment across backends.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from declearn.dataset.examples import load_mnist
from declearn.dataset.utils import split_multi_classif_dataset

from benchmarks.workload import baseline as B

__all__ = [
    "BENCH_ROOT",
    "DATA_ROOT",
    "ensure_data_for_n_clients",
    "ensure_source_data",
]


SEED = 42

BENCH_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = BENCH_ROOT / "data"

_VALID_LAYOUTS = ("chw", "hwc")


def _source_dir(n_clients: int) -> Path:
    return DATA_ROOT / f"source_{n_clients}"


def _fraction_tag(fraction: float) -> str:
    """Stable directory suffix for a sample-fraction value.

    Used so that cached layout dirs at different fractions don't
    collide. `1.0` -> `"f100"`, `0.1` -> `"f010"`, etc.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"fraction must be in (0, 1]; got {fraction!r}."
        )
    return f"f{int(round(fraction * 100)):03d}"


def _layout_dir(n_clients: int, layout: str, fraction: float) -> Path:
    return DATA_ROOT / f"{layout}_{n_clients}_{_fraction_tag(fraction)}"


def _client_files(folder: Path, idx: int) -> Tuple[Path, Path, Path, Path]:
    sub = folder / f"client_{idx}"
    return (
        sub / "train_data.npy",
        sub / "train_target.npy",
        sub / "valid_data.npy",
        sub / "valid_target.npy",
    )


def _is_complete(folder: Path, n_clients: int) -> bool:
    if not folder.is_dir():
        return False
    for idx in range(n_clients):
        for path in _client_files(folder, idx):
            if not path.is_file():
                return False
    return True


def ensure_source_data(n_clients: int) -> Path:
    """Produce or return the canonical IID MNIST split for `n_clients`.

    The split is materialized as `data/source_<n_clients>/client_<i>/`
    folders, each holding `train_data.npy`, `train_target.npy`, and
    their `valid_*` counterparts. Image arrays are stored as float32
    of shape `(N, 28, 28)` and targets as uint8 of shape `(N,)`.

    Idempotent: subsequent calls with the same `n_clients` are no-ops.
    """
    folder = _source_dir(n_clients)
    if _is_complete(folder, n_clients):
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    images, labels = load_mnist(train=True, folder=str(DATA_ROOT / "_mnist"))
    shards = split_multi_classif_dataset(
        dataset=(images, labels),
        n_shards=n_clients,
        scheme="iid",
        p_valid=0.2,
        seed=SEED,
    )
    for idx, ((x_train, y_train), (x_valid, y_valid)) in enumerate(shards):
        sub = folder / f"client_{idx}"
        sub.mkdir(parents=True, exist_ok=True)
        np.save(sub / "train_data.npy", x_train)
        np.save(sub / "train_target.npy", y_train)
        np.save(sub / "valid_data.npy", x_valid)
        np.save(sub / "valid_target.npy", y_valid)
    return folder


def _convert(array: np.ndarray, layout: str, is_target: bool) -> np.ndarray:
    if is_target:
        return array
    if layout == "chw":
        # (N, 28, 28) -> (N, 1, 28, 28)
        return array.reshape(array.shape[0], 1, 28, 28).astype(np.float32)
    if layout == "hwc":
        # (N, 28, 28) -> (N, 28, 28, 1)
        return array.reshape(array.shape[0], 28, 28, 1).astype(np.float32)
    raise ValueError(
        f"Unknown layout '{layout}'. Expected one of {_VALID_LAYOUTS}."
    )


def _slice(array: np.ndarray, fraction: float) -> np.ndarray:
    """Return the leading `fraction` of `array` along axis 0."""
    if fraction >= 1.0:
        return array
    n = max(1, int(round(array.shape[0] * fraction)))
    return array[:n]


def ensure_data_for_n_clients(
    n_clients: int,
    layout: str,
    fraction: float = B.BASELINE_DATASET_FRACTION,
) -> Path:
    """Produce or return the requested layout for `n_clients`.

    Layouts:
        - "chw":  (N, 1, 28, 28) float32, source uint8 targets (torch)
        - "hwc":  (N, 28, 28, 1) float32, source uint8 targets (TF)

    `fraction` slices the leading prefix of each client's train/valid
    arrays — e.g. `fraction=0.1` keeps 10 % of each shard. The cache
    folder name embeds the fraction so different fractions coexist on
    disk. Re-derives from the canonical source split when missing.
    """
    if layout not in _VALID_LAYOUTS:
        raise ValueError(
            f"Unknown layout '{layout}'. Expected one of {_VALID_LAYOUTS}."
        )
    folder = _layout_dir(n_clients, layout, fraction)
    if _is_complete(folder, n_clients):
        return folder
    source = ensure_source_data(n_clients)
    folder.mkdir(parents=True, exist_ok=True)
    for idx in range(n_clients):
        src_train_d, src_train_t, src_valid_d, src_valid_t = _client_files(
            source, idx
        )
        dst_train_d, dst_train_t, dst_valid_d, dst_valid_t = _client_files(
            folder, idx
        )
        dst_train_d.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            dst_train_d,
            _convert(_slice(np.load(src_train_d), fraction), layout, False),
        )
        np.save(
            dst_train_t,
            _convert(_slice(np.load(src_train_t), fraction), layout, True),
        )
        np.save(
            dst_valid_d,
            _convert(_slice(np.load(src_valid_d), fraction), layout, False),
        )
        np.save(
            dst_valid_t,
            _convert(_slice(np.load(src_valid_t), fraction), layout, True),
        )
    return folder


def client_data_paths(
    n_clients: int,
    layout: str,
    idx: int,
    fraction: float = B.BASELINE_DATASET_FRACTION,
) -> Tuple[Path, Path, Path, Path]:
    """Return the four per-client array paths for a prepared layout."""
    folder = _layout_dir(n_clients, layout, fraction)
    return _client_files(folder, idx)
