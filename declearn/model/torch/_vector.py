# coding: utf-8

"""TorchVector gradients container."""

from typing import Any, Callable, Dict, Set, Tuple, Type

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
