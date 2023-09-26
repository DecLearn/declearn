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

"""Implementation of `build_samplewise_grads_fn` for Torch 2.0."""

from typing import List, Tuple

import functorch  # type: ignore
import torch

from declearn.model.torch._samplewise.shared import (
    GetGradientsFunction,
    clip_and_scale_grads_inplace,
)

__all__ = [
    "build_samplewise_grads_fn_backend",
]


def build_samplewise_grads_fn_backend(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    inputs: int,
    y_true: bool,
    s_wght: bool,
) -> GetGradientsFunction:
    """Implementation of `build_samplewise_grads_fn` for Torch 1.1X."""

    func_model, *_ = functorch.make_functional_with_buffers(model)

    def run_forward(inputs, y_true, s_wght, buffers, *params):
        """Run the forward pass in a functional way."""
        y_pred = func_model(params, buffers, *inputs)
        s_loss = loss_fn(y_pred, y_true)
        if s_wght is not None:
            s_loss.mul_(s_wght.to(s_loss.device))
        return s_loss.mean()

    def grads_fn(inputs, y_true, s_wght, clip=None):
        """Compute gradients and optionally clip them."""
        params, idxgrd, pnames = get_params(model)
        buffers = list(model.buffers())
        gfunc = functorch.grad_and_value(run_forward, argnums=tuple(idxgrd))
        grads, loss = gfunc(
            inputs, y_true, (None if clip else s_wght), buffers, *params
        )
        if clip:
            clip_and_scale_grads_inplace(grads, clip, s_wght)
        return dict(zip(pnames, grads)), loss.detach()

    # Wrap the former function to compute and clip sample-wise gradients.
    in_dims = ([0] * inputs, 0 if y_true else None, 0 if s_wght else None)
    return functorch.vmap(grads_fn, in_dims, randomness="same")


def get_params(
    model: torch.nn.Module,
) -> Tuple[List[torch.nn.Parameter], List[int], List[str]]:
    """Return a model's parameters and the index and name of trainable ones."""
    params = []  # type: List[torch.nn.Parameter]
    idxgrd = []  # type: List[int]
    pnames = []  # type: List[str]
    for idx, (name, param) in enumerate(model.named_parameters()):
        params.append(param)
        if param.requires_grad:
            idxgrd.append(idx + 4)
            pnames.append(name)
    return params, idxgrd, pnames
