# coding: utf-8

"""TorchVector gradients container."""

from typing import Any, Callable, Dict

import numpy as np
import torch
from typing_extensions import Self  # future: import from typing (Py>=3.11)

from declearn2.model.api import NumpyVector, Vector, register_vector_type


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

    def __init__(
            self,
            coefs: Dict[str, torch.Tensor]
        ) -> None:
        super().__init__(coefs)

    def pack(
            self,
        ) -> Dict[str, Any]:
        return {key: tns.numpy() for key, tns in self.coefs.items()}

    @classmethod
    def unpack(
            cls,
            data: Dict[str, Any],
        ) -> 'TorchVector':
        # false-positive; pylint: disable=no-member
        coef = {key: torch.from_numpy(dat) for key, dat in data.items()}
        return cls(coef)

    def _apply_operation(
            self,
            other: Any,
            func: Callable[[Any, Any], Any],
        ) -> Self:  # type: ignore
        # Extend support to (TensorflowVector, NumpyVector) combinations.
        if isinstance(other, NumpyVector):
            if self.coefs.keys() != other.coefs.keys():
                raise KeyError(
                    f"Cannot {func.__name__} Vectors "\
                    "with distinct coefficient names."
                )
            return type(self)({
                key: func(self.coefs[key], other.coefs[key])
                for key in self.coefs
            })
        # Delegate other cases to parent class.
        return super()._apply_operation(other, func)

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

    def sign(
            self
        ) -> Self:  # type: ignore
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
