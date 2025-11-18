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

"""Shared code for torch-version-dependent backend code."""

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

__all__ = [
    "GetGradientsFunction",
    "clip_and_scale_grads_inplace",
]


GetGradientsFunction = Callable[
    [
        List[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[float],
    ],
    Tuple[Dict[str, torch.Tensor], torch.Tensor],
]
"""Signature for sample-wise gradients computation functions."""


def clip_and_scale_grads_inplace(
    grads: Iterable[torch.Tensor],
    clip: float,
    wght: Optional[torch.Tensor] = None,
) -> None:
    """Clip a collection of tensors in-place, based on their euclidean norm.

    Also apply an optional weight tensor to scale the clipped gradients.
    """
    for grad in grads:
        norm = torch.norm(grad, p=2, keepdim=True)
        grad.mul_(torch.clamp(clip / norm, max=1))
        if wght is not None:
            grad.mul_(wght.to(grad.device))
