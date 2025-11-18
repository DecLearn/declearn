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

"""Backend utils for metrics' computations."""

from typing import Tuple

import numpy as np

__all__ = [
    "safe_division",
    "squeeze_into_identical_shapes",
]


def squeeze_into_identical_shapes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Verify that inputs have identical shapes, up to squeezable dims.

    Return the input arrays, squeezed when needed.
    Raise a ValueError if they cannot be made to match.
    """
    # Case of identical-shape inputs.
    if y_true.shape == y_pred.shape:
        return y_true, y_pred
    # Case of identical-shape inputs up to squeezable dims.
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    if y_true.shape == y_pred.shape:
        # Handle edge case of scalar values: preserve one dimension.
        if not y_true.shape:
            y_true = np.expand_dims(y_true, 0)
            y_pred = np.expand_dims(y_pred, 0)
        return y_true, y_pred
    # Case of mismatching shapes.
    raise ValueError(
        "Received inputs with incompatible shapes: "
        f"y_true has shape {y_true.shape}, y_pred has shape {y_pred.shape}."
    )


def safe_division(
    num: float,
    den: float,
    default: float = 0.0,
) -> float:
    """Perform a division, avoiding ZeroDivisionError in favor of a default."""
    if den == 0.0:
        return default
    return num / den
