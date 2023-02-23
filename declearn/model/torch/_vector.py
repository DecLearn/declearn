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
from declearn.model.torch.utils import select_device
from declearn.utils import get_device_policy


__all__ = [
    "TorchVector",
]


@register_vector_type(torch.Tensor)
class TorchVector(Vector):
    """Vector subclass to store PyTorch tensors.

    This Vector is designed to store a collection of named PyTorch
    tensors, enabling computations that are either applied to each
    and every coefficient, or imply two sets of aligned coefficients
    (i.e. two TorchVector with similar specifications).

    Use `vector.coefs` to access the stored coefficients.

    Notes
    -----
    - A `TorchVector` can be operated with either a:
      - scalar value
      - `NumpyVector` that has similar specifications
      - `TorchVector` that has similar specifications
      => resulting in a `TorchVector` in each of these cases.
    - The wrapped tensors may be placed on any device (CPU, GPU...)
      and may not be all on the same device.
    - The device-placement of the initial `TorchVector`'s data
      is preserved by operations, including with `NumpyVector`.
    - When combining two `TorchVector`, the device-placement
      of the left-most one is used; in that case, one ends up with
      `gpu + cpu = gpu` while `cpu + gpu = cpu`. In both cases, a
      warning will be emitted to prevent silent un-optimized copies.
    - When deserializing a `TorchVector` (either by directly using
      `TorchVector.unpack` or loading one from a JSON dump), loaded
      tensors are placed based on the global device-placement policy
      (accessed via `declearn.utils.get_device_policy`). Thus it may
      have a different device-placement schema than at dump time but
      should be coherent with that of `TorchModel` computations.
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
        # Convert 'other' NumpyVector to a (CPU-backed) TorchVector.
        if isinstance(other, NumpyVector):
            # false-positive; pylint: disable=no-member
            coefs = {
                key: torch.from_numpy(val) for key, val in other.coefs.items()
            }
            other = TorchVector(coefs)
        # Ensure 'other' TorchVector shares this vector's device placement.
        if isinstance(other, TorchVector):
            coefs = {
                key: val.to(self.coefs[key].device)
                for key, val in other.coefs.items()
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
        return {
            key: np.array(tns.cpu().numpy()) for key, tns in self.coefs.items()
        }

    @classmethod
    def unpack(
        cls,
        data: Dict[str, Any],
    ) -> Self:
        policy = get_device_policy()
        device = select_device(gpu=policy.gpu, idx=policy.idx)
        coefs = {
            # false-positive on `torch.from_numpy`; pylint: disable=no-member
            key: torch.from_numpy(dat).to(device)
            for key, dat in data.items()
        }
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
                # false-positive on 'torch.equal'; pylint: disable=no-member
                torch.equal(tns, other.coefs[key].to(tns.device))
                for key, tns in self.coefs.items()
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
