"""Side-effect memory probes for the declearn benchmark suite.

`time_run` calls `capture_start` before the workload and `capture_end`
after, writing peak host-RSS-delta and peak GPU bytes to a per-cell
`/tmp` cache (microseconds of overhead; invisible against the
~70-180s workload). The `track_peakmem_run` / `track_peakgpu_run`
methods then `read_cached` those numbers without re-running the
workload, so adding memory tracking does not lengthen the sweep.

The delta-RSS form (peak minus baseline) sidesteps the "setup dominates
peak" blind spot of raw `ru_maxrss`; the GPU number sidesteps
`ru_maxrss` missing VRAM entirely.

Helper names here intentionally avoid ASV's benchmark name patterns
(`time_*`, `mem_*`, `peakmem_*`, `track_*`), so ASV does not mistake
them for benchmarks when it walks this module.
"""

import json
import os
import resource
import tempfile
from typing import Tuple

import torch

__all__ = ["capture_start", "capture_end", "read_cached"]


def _cuda() -> bool:
    return torch.cuda.is_available()


def _cache_path(cls_name: str, params: Tuple) -> str:
    """Return the per-cell `/tmp` cache path for a class/params combo."""
    key = "_".join(str(p) for p in params)
    return os.path.join(
        tempfile.gettempdir(), f"declearn_mem_{cls_name}_{key}.json"
    )


def capture_start() -> int:
    """Reset GPU peak stats and return the baseline host RSS (KiB)."""
    if _cuda():
        torch.cuda.reset_peak_memory_stats()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def capture_end(cls_name: str, params: Tuple, baseline_kb: int) -> None:
    """Cache peak host-RSS-delta and peak GPU bytes for this cell."""
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    host_delta_bytes = max(peak_kb - baseline_kb, 0) * 1024
    gpu_bytes = torch.cuda.max_memory_allocated() if _cuda() else 0
    with open(_cache_path(cls_name, params), "w") as f:
        json.dump(
            {"host_delta_bytes": host_delta_bytes, "gpu_bytes": gpu_bytes}, f
        )


def read_cached(cls_name: str, params: Tuple, key: str) -> int:
    """Return a cached metric for this cell, or 0 if absent/corrupt."""
    try:
        with open(_cache_path(cls_name, params)) as f:
            return int(json.load(f).get(key, 0))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0
