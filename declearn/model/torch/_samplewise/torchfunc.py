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

"""Implementation of `build_samplewise_grads_fn` for Torch 2.0."""

from typing import Dict, Tuple

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
    """Implementation of `build_samplewise_grads_fn` for Torch 2.0."""

    # pylint: disable-next=too-many-positional-arguments
    def run_forward(params, frozen, buffers, inputs, y_true, s_wght):  # noqa: PLR0913
        """Run the forward pass in a functional way."""
        # backend closure function; pylint: disable=too-many-arguments
        y_pred = torch.func.functional_call(
            model, [params, frozen, buffers], *inputs
        )
        s_loss = loss_fn(y_pred, y_true)
        if s_wght is not None:
            s_loss.mul_(s_wght.to(s_loss.device))
        return s_loss.mean()

    get_grads_and_loss = torch.func.grad_and_value(run_forward, argnums=0)

    def get_clipped_grads_and_loss(inputs, y_true, s_wght, clip=None):
        """Compute gradients and optionally clip them."""
        params, frozen = get_params(model)
        buffers = dict(model.named_buffers())
        grads, loss = get_grads_and_loss(
            params, frozen, buffers, inputs, y_true, None if clip else s_wght
        )
        if clip:
            clip_and_scale_grads_inplace(grads.values(), clip, s_wght)
        return grads, loss.detach()

    # Wrap the former function to compute and clip sample-wise gradients.
    in_dims = ([0] * inputs, 0 if y_true else None, 0 if s_wght else None)
    return torch.func.vmap(
        get_clipped_grads_and_loss, in_dims, randomness="same"
    )


def get_params(
    model: torch.nn.Module,
) -> Tuple[Dict[str, torch.nn.Parameter], Dict[str, torch.nn.Parameter]]:
    """Return a model's parameters, split between trainable and frozen ones."""
    params: Dict[str, torch.nn.Parameter] = {}
    frozen: Dict[str, torch.nn.Parameter] = {}
    for name, param in model.named_parameters():
        (params if param.requires_grad else frozen)[name] = param
    return params, frozen
