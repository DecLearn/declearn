# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""Utils for GPU support and device management in tensorflow."""

import functools
import warnings
from typing import Any, Callable, Optional, Union

import jax
import jaxlib.xla_extension as xe

__all__ = ["select_device"]


def select_device(
    gpu: bool,
    idx: Optional[int] = None,
) -> xe.Device:
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
    UserWarning:
        If `gpu=True` but no GPU is available.
        If `idx` exceeds the number of available GPU devices.

    Returns
    -------
    device: tf.config.LogicalDevice
        Selected device, usable as `tf.device` argument.
    """
    idx = 0 if idx is None else idx
    # List available CPU or GPU devices.
    device_type = "gpu" if gpu else "cpu"
    devices = [d for d in jax.devices() if d.platform == device_type]
    # Case when no GPU is available: warn and use a CPU instead.
    if gpu and not devices:
        warnings.warn(
            "Cannot use a GPU device: either CUDA is unavailable "
            "or no GPU is visible to jax."
        )
        idx = 0
        devices = jax.devices()
    # Case when the desired device index is invalid: select another one.
    if idx >= len(devices):
        warnings.warn(
            f"Cannot use {device_type} device n°{idx}: index is out-of-range."
            f"\nUsing {device_type} device n°0 instead."
        )
        idx = 0
    # Return the selected device.
    return devices[idx]
