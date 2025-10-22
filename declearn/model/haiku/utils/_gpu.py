# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
# et Automatique)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for GPU support and device management in jax."""

import warnings
from typing import Optional

import jax

__all__ = ["select_device"]


def select_device(
    gpu: bool,
    idx: Optional[int] = None,
) -> jax.Device:
    """Select a backing device to use based on inputs and availability.

    Parameters
    ----------
    gpu: bool
        Whether to select a GPU device rather than the CPU one.
    idx: int or None, default=None
        Optional pre-selected device index. Only used when `gpu=True`.
        If `idx is None` or exceeds the number of available GPU devices,
        use the first available one.

    Warns
    -----
    RuntimeWarning:
        If `gpu=True` but no GPU is available.
        If `idx` exceeds the number of available GPU devices.

    Returns
    -------
    device: jax.Device
        Selected device.
    """
    idx = 0 if idx is None else idx
    device_type = "gpu" if gpu else "cpu"
    # List devices, handling errors related to the lack of GPU (or CPU error).
    try:
        devices = jax.devices(device_type)
    except RuntimeError as exc:
        # Warn about the lack of GPU (support?), and fall back to CPU.
        if gpu:
            warnings.warn(
                "Cannot use a GPU device: either CUDA is unavailable "
                f"or no GPU is visible to jax: raised {repr(exc)}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return select_device(gpu=False, idx=0)
        # Case when no CPU is found: this should never be reached.
        raise RuntimeError(  # pragma: no cover
            "Failed to have jax select a CPU device."
        ) from exc
    # similar code to tensorflow util; pylint: disable=duplicate-code
    # Case when the desired device index is invalid: select another one.
    if idx >= len(devices):
        warnings.warn(
            f"Cannot use {device_type} device n°{idx}: index is out-of-range."
            f"\nUsing {device_type} device n°0 instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        idx = 0
    # Return the selected device.
    return devices[idx]
