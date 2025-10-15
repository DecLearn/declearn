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

"""Utilities to convert to and from numpy, commonly used in declearn tests."""

import importlib
from typing import Any

import numpy as np

__all__ = ["to_numpy"]


def to_numpy(array: Any, framework: str) -> np.ndarray:
    """Convert an input framework-based structure to a numpy array."""
    if isinstance(array, np.ndarray):
        return array
    if framework == "jax":
        return np.asarray(array)
    if framework == "tensorflow":  # add support for IndexedSlices
        tensorflow = importlib.import_module("tensorflow")
        if isinstance(array, tensorflow.IndexedSlices):
            with tensorflow.device(array.device):
                return tensorflow.convert_to_tensor(array).numpy()
        return array.numpy()
    if framework == "torch":
        return array.cpu().numpy()
    raise ValueError(
        f"Invalid 'framework' from which to convert to numpy: '{framework}'."
    )
