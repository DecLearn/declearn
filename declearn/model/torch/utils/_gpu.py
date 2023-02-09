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

"""Utils for GPU support and device management in torch."""

import warnings
from typing import Any, Optional

import torch


__all__ = [
    "AutoDeviceModule",
    "select_device",
]


def select_device(
    gpu: bool,
    idx: Optional[int] = None,
) -> torch.device:  # pylint: disable=no-member
    """Select a backing device to use based on inputs and availability.

    Parameters
    ----------
    gpu: bool
        Whether to select a GPU device rather than the CPU one.
    idx: int or None, default=None
        Optional pre-selected GPU device index. Only used when `gpu=True`.
        If `idx is None` or exceeds the number of available GPU devices,
        use `torch.cuda.current_device()`.

    Warns
    -----
    UserWarning:
        If `gpu=True` but no GPU is available.
        If `idx` exceeds the number of available GPU devices.

    Returns
    -------
    device: torch.device
        Selected torch device, with type "cpu" or "cuda".
    """
    # Case when instructed to use the CPU device.
    if not gpu:
        return torch.device("cpu")  # pylint: disable=no-member
    # Case when no GPU is available: warn and use the CPU instead.
    if gpu and not torch.cuda.is_available():
        warnings.warn(
            "Cannot use a GPU device: either CUDA is unavailable "
            "or no GPU is visible to torch."
        )
        return torch.device("cpu")  # pylint: disable=no-member
    # Case when the desired GPU is invalid: select another one.
    if (idx or 0) >= torch.cuda.device_count():
        warnings.warn(
            f"Cannot use GPU device n°{idx}: index is out-of-range.\n"
            f"Using GPU device n°{torch.cuda.current_device()} instead."
        )
        idx = None
    # Return the selected or auto-selected GPU device index.
    if idx is None:
        idx = torch.cuda.current_device()
    return torch.device("cuda", index=idx)  # pylint: disable=no-member


class AutoDeviceModule(torch.nn.Module):
    """Wrapper for a `torch.nn.Module`, automating device-management.

    This `torch.nn.Module` subclass enables wrapping another one, and
    provides:
    * a `device` attribute (and instantiation parameter) indicating
      where the wrapped module is placed
    * automatic placement of input tensors on that device as part of
      `forward` calls to the module
    * a `set_device` method to change the device and move the wrapped
      module to it

    This aims at internalizing device-management boilerplate code.
    The wrapped module is assigned to the `module` attribute and thus
    can be accessed directly.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        device: torch.device,  # pylint: disable=no-member
    ) -> None:
        """Wrap a torch Module into an AutoDeviceModule.

        Parameters
        ----------
        module: torch.nn.Module
            Torch module that needs wrapping.
        device: torch.device
            Torch device where to place the wrapped module and computations.
        """
        super().__init__()
        self.device = device
        self.module = module.to(self.device)

    def forward(self, *inputs: Any) -> torch.Tensor:
        """Run the forward computation, automating device-placement of inputs.

        Please refer to `self.module.forward` for details on the wrapped
        module's forward specifications.
        """
        inputs = tuple(
            x.to(self.device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        )
        return self.module(*inputs)

    def set_device(
        self,
        device: torch.device,  # pylint: disable=no-member
    ) -> None:
        """Move the wrapped module to a pre-selected torch device.

        Parameters
        ----------
        device: torch.device
           Torch device where to place the wrapped module and computations.
        """
        self.device = device
        self.module.to(device)
