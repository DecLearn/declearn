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

import tensorflow as tf  # type: ignore


__all__ = [
    "move_layer_to_device",
    "preserve_tensor_device",
    "select_device",
]


def select_device(
    gpu: bool,
    idx: Optional[int] = None,
) -> tf.config.LogicalDevice:
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
    device_type = "GPU" if gpu else "CPU"
    devices = tf.config.list_logical_devices(device_type)
    # Case when no GPU is available: warn and use a CPU instead.
    if gpu and not devices:
        warnings.warn(
            "Cannot use a GPU device: either CUDA is unavailable "
            "or no GPU is visible to tensorflow."
        )
        device_type, idx = "CPU", 0
        devices = tf.config.list_logical_devices("CPU")
    # Case when the desired device index is invalid: select another one.
    if idx >= len(devices):
        warnings.warn(
            f"Cannot use {device_type} device n°{idx}: index is out-of-range."
            f"\nUsing {device_type} device n°0 instead."
        )
        idx = 0
    # Return the selected device.
    return devices[idx]


def move_layer_to_device(
    layer: tf.keras.layers.Layer,
    device: Union[tf.config.LogicalDevice, str],
) -> tf.keras.layers.Layer:
    """Create a copy of an input keras layer placed on a given device.

    This functions creates a copy of the input layer and of all its weights.
    It may therefore be costful and should be used sparingly, to move away
    variables on a device where all further computations are expected to be
    run.

    Parameters
    ----------
    layer: tf.keras.layers.Layer
        Keras layer that needs moving to another device.
    device: tf.config.LogicalDevice or str
        Device where to place the layer's weights.

    Returns
    -------
    layer: tf.keras.layers.Layer
        Copy of the input layer, with its weights backed on `device`.
    """
    config = tf.keras.layers.serialize(layer)
    weights = layer.get_weights()
    with tf.device(device):
        layer = tf.keras.layers.deserialize(config)
        layer.set_weights(weights)
    return layer


def preserve_tensor_device(
    func: Callable[..., tf.Tensor],
) -> Callable[..., tf.Tensor]:
    """Wrap a tensor-processing function to have it run on its inputs' device.

    Parameters
    ----------
    func: function(tf.Tensor, ...) -> tf.Tensor:
        Function to wrap, that takes a tensorflow Tensor as first argument.

    Returns
    -------
    func: function(tf.Tensor, ...) -> tf.Tensor:
        Similar function to the input one, that operates under a `tf.device`
        context so as to run computations on the first input tensor's device.
    """

    @functools.wraps(func)
    def wrapped(tensor: tf.Tensor, *args: Any, **kwargs: Any) -> tf.Tensor:
        """Wrapped function, running under a `tf.device` context."""
        with tf.device(tensor.device):
            return func(tensor, *args, **kwargs)

    return wrapped
