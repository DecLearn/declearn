# coding: utf-8

"""TorchVector gradients container."""

from typing import Any, Dict, Set, Type

import numpy as np
import torch
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn.model.api import NumpyVector, Vector, register_vector_type


@register_vector_type(torch.Tensor)
class TorchVector(Vector):
    """Vector subclass to store PyTorch tensors.

    This Vector is designed to store a collection of named
    PyTorch tensors, enabling computations that are either
    applied to each and every coefficient, or imply two sets
    of aligned coefficients (i.e. two TorchVector with
    similar specifications).
    """

    _op_add = torch.add  # pylint: disable=no-member
    _op_sub = torch.sub  # pylint: disable=no-member
    _op_mul = torch.mul  # pylint: disable=no-member
    _op_div = torch.div  # pylint: disable=no-member
    _op_pow = torch.pow  # pylint: disable=no-member

    @property
    def compatible_vector_types(self) -> Set[Type[Vector]]:
        types = super().compatible_vector_types
        return types.union({NumpyVector, TorchVector})

    def __init__(self, coefs: Dict[str, torch.Tensor]) -> None:
        super().__init__(coefs)

    def __repr__(
        self,
    ) -> str:
        string = f"{type(self).__name__} with {len(self.coefs)} coefs:"
        string += "".join(
            f"\n    {key}: {val.dtype} tensor with shape {val.shape}"
            for key, val in self.coefs.items()
        )
        return string

    def pack(
        self,
    ) -> Dict[str, Any]:
        return {key: tns.numpy() for key, tns in self.coefs.items()}

    @classmethod
    def unpack(
        cls,
        data: Dict[str, Any],
    ) -> "TorchVector":
        # false-positive; pylint: disable=no-member
        coefs = {key: torch.from_numpy(dat) for key, dat in data.items()}
        return cls(coefs)

    def __eq__(
        self,
        other: Any,
    ) -> bool:
        valid = isinstance(other, TorchVector)
        valid = valid and (self.coefs.keys() == other.coefs.keys())
        return valid and all(
            np.array_equal(self.coefs[k].numpy(), other.coefs[k].numpy())
            for k in self.coefs
        )

    def sign(self) -> Self:  # type: ignore
        # false-positive; pylint: disable=no-member
        return self.apply_func(torch.sign)

    def minimum(
        self,
        other: Any,
    ) -> Self:  # type: ignore
        # false-positive; pylint: disable=no-member
        if isinstance(other, Vector):
            return self._apply_operation(other, torch.minimum)
        return self.apply_func(torch.minimum, other)

    def maximum(
        self,
        other: Any,
    ) -> Self:  # type: ignore
        # false-positive; pylint: disable=no-member
        if isinstance(other, Vector):
            return self._apply_operation(other, torch.minimum)
        return self.apply_func(torch.maximum, other)
