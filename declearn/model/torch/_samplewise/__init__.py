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

"""Torch-version-dependent code to compute sample-wise gradients."""

from typing import Callable, Dict, List, Optional

import torch

from .shared import GetGradientsFunction

if torch.__version__.startswith("2."):
    from .torchfunc import build_samplewise_grads_fn_backend
else:
    # pragma: no cover
    raise ImportError(f"Unsupported Torch version: {torch.__version__}")


__all__ = [
    "GetGradientsFunction",
    "build_samplewise_grads_fn",
]


def build_samplewise_grads_fn(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    inputs: int,
    y_true: bool,
    s_wght: bool,
) -> GetGradientsFunction:
    """Build a torch-specific sample-wise gradients-computation function.

    Parameters
    ----------
    model: torch.nn.Module
        Model that is to be trained.
    loss_fn: torch.nn.Module
        Loss-computing module, returning sample-wise loss values.
    inputs: int
        Number of input tensors.
    y_true: bool
        Whether a true labels tensor is provided.
    s_wght: bool
        Whether a sample weights tensor is provided.

    Returns
    -------
    grads_fn: callable[[inputs, y_true, s_wght, clip], (grads, loss)]
        Function that efficiently computes and returns sample-wise gradients
        wrt trainable model parameters based on a batch of inputs, with opt.
        clipping based on a maximum l2-norm value `clip`.
        It returns the sample-wise gradients as a dict of tensors with their
        parameter name as key, plus the sample-wise loss values as a tensor.
    """
    return build_samplewise_grads_fn_backend(
        model, loss_fn, inputs, y_true, s_wght
    )
