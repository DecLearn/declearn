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

"""TorchVector data arrays container."""

from typing import Any, Callable, Dict, Optional, Set, Tuple, Type

import numpy as np
import torch
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn.model.api import Vector, register_vector_type
from declearn.model.sklearn import NumpyVector


@register_vector_type(torch.Tensor)
class TorchVector(Vector):
    """Vector subclass to store PyTorch tensors.

    This Vector is designed to store a collection of named
    PyTorch tensors, enabling computations that are either
    applied to each and every coefficient, or imply two sets
    of aligned coefficients (i.e. two TorchVector with
    similar specifications).

    Use `vector.coefs` to access the stored coefficients.
    """

    @property
    def _op_add(self) -> Callable[[Any, Any], Any]:
        return torch.add  # pylint: disable=no-member

    @property
    def _op_sub(self) -> Callable[[Any, Any], Any]:
        return torch.sub  # pylint: disable=no-member

    @property
    def _op_mul(self) -> Callable[[Any, Any], Any]:
        return torch.mul  # pylint: disable=no-member

    @property
    def _op_div(self) -> Callable[[Any, Any], Any]:
        return torch.div  # pylint: disable=no-member

    @property
    def _op_pow(self) -> Callable[[Any, Any], Any]:
        return torch.pow  # pylint: disable=no-member

    @property
    def compatible_vector_types(self) -> Set[Type[Vector]]:
        types = super().compatible_vector_types
        return types.union({NumpyVector, TorchVector})

    def __init__(self, coefs: Dict[str, torch.Tensor]) -> None:
        super().__init__(coefs)

    def _apply_operation(
        self,
        other: Any,
        func: Callable[[Any, Any], Any],
    ) -> Self:
        if isinstance(other, NumpyVector):
            # false-positive; pylint: disable=no-member
            coefs = {
                key: torch.from_numpy(val) for key, val in other.coefs.items()
            }
            other = TorchVector(coefs)
        return super()._apply_operation(other, func)

    def dtypes(
        self,
    ) -> Dict[str, str]:
        dtypes = super().dtypes()
        return {key: val.split(".", 1)[-1] for key, val in dtypes.items()}

    def shapes(
        self,
    ) -> Dict[str, Tuple[int, ...]]:
        return {key: tuple(coef.shape) for key, coef in self.coefs.items()}

    def pack(
        self,
    ) -> Dict[str, Any]:
        return {key: tns.numpy() for key, tns in self.coefs.items()}

    @classmethod
    def unpack(
        cls,
        data: Dict[str, Any],
    ) -> Self:
        # false-positive; pylint: disable=no-member
        coefs = {key: torch.from_numpy(dat) for key, dat in data.items()}
        return cls(coefs)

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        valid = isinstance(other, TorchVector)
        if valid:
            valid = self.coefs.keys() == other.coefs.keys()
        if valid:
            valid = all(
                np.array_equal(self.coefs[k].numpy(), other.coefs[k].numpy())
                for k in self.coefs
            )
        return valid

    def sign(self) -> Self:
        # false-positive; pylint: disable=no-member
        return self.apply_func(torch.sign)

    def minimum(
        self,
        other: Any,
    ) -> Self:
        # false-positive; pylint: disable=no-member
        if isinstance(other, Vector):
            return self._apply_operation(other, torch.minimum)
        if isinstance(other, float):
            other = torch.Tensor([other])
        return self.apply_func(torch.minimum, other)

    def maximum(
        self,
        other: Any,
    ) -> Self:
        # false-positive; pylint: disable=no-member
        if isinstance(other, Vector):
            return self._apply_operation(other, torch.maximum)
        if isinstance(other, float):
            other = torch.Tensor([other])
        return self.apply_func(torch.maximum, other)

    def sum(
        self,
        axis: Optional[int] = None,
        keepdims: bool = False,
    ) -> Self:
        coefs = {
            key: val.sum(dim=axis, keepdims=keepdims)
            for key, val in self.coefs.items()
        }
        return self.__class__(coefs)
