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

"""NumpyVector data arrays container."""

import warnings
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn.model.api._vector import Vector, register_vector_type


__all__ = [
    "NumpyVector",
]


@register_vector_type(np.ndarray)
class NumpyVector(Vector):
    """Vector subclass to store numpy.ndarray coefficients.

    This Vector is designed to store a collection of named
    numpy arrays or scalars, enabling computations that are
    either applied to each and every coefficient, or imply
    two sets of aligned coefficients (i.e. two NumpyVector
    instances with similar coefficients specifications).

    Use `vector.coefs` to access the stored coefficients.

    Notes
    -----
    - A `NumpyVector` can be operated with either a scalar value,
      or another `NumpyVector` that has similar specifications
      (same coefficient names, shapes and compatible dtypes).
    - Some other `Vector` classes might be made compatible with
      `NumpyVector`; in that case, operating with a `NumpyVector`
      will always result in a vector of the other type. This is
      notably the case with `TensorflowVector` and `TorchVector`.
    - There is currently no support for GPU-acceleration with the
      `NumpyVector` class, that only handles arrays and operations
      placed on a CPU device.
    """

    @property
    def _op_add(self) -> Callable[[Any, Any], np.ndarray]:
        return np.add

    @property
    def _op_sub(self) -> Callable[[Any, Any], np.ndarray]:
        return np.subtract

    @property
    def _op_mul(self) -> Callable[[Any, Any], np.ndarray]:
        return np.multiply

    @property
    def _op_div(self) -> Callable[[Any, Any], np.ndarray]:
        return np.divide

    @property
    def _op_pow(self) -> Callable[[Any, Any], np.ndarray]:
        return np.power

    def __init__(
        self,
        coefs: Dict[str, np.ndarray],
    ) -> None:
        super().__init__(coefs)

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        valid = isinstance(other, NumpyVector)
        if valid:
            valid = self.coefs.keys() == other.coefs.keys()
        if valid:
            valid = all(
                np.array_equal(self.coefs[k], other.coefs[k])
                for k in self.coefs
            )
        return valid

    def sign(
        self,
    ) -> Self:
        return self.apply_func(np.sign)

    def minimum(
        self,
        other: Union[Self, float],
    ) -> Self:
        if isinstance(other, NumpyVector):
            return self._apply_operation(other, np.minimum)
        return self.apply_func(np.minimum, other)

    def maximum(
        self,
        other: Union[Self, float],
    ) -> Self:
        if isinstance(other, Vector):
            return self._apply_operation(other, np.maximum)
        return self.apply_func(np.maximum, other)

    def sum(
        self,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> Self:
        if isinstance(axis, int) or keepdims:
            warnings.warn(  # pragma: no cover
                "The 'axis' and 'keepdims' arguments of 'NumpyVector.sum' "
                "have been deprecated as of declearn v2.3, and will be "
                "removed in version 2.5 and/or 3.0.",
                DeprecationWarning,
            )
        coefs = {
            key: np.array(np.sum(val, axis=axis, keepdims=keepdims))
            for key, val in self.coefs.items()
        }
        return self.__class__(coefs)
