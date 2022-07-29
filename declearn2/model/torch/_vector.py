# coding: utf-8

"""TorchVector gradients container."""

import json
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from declearn2.model.api import NumpyVector, Vector, register_vector_type
from declearn2.utils import deserialize_numpy, serialize_numpy


@register_vector_type(torch.Tensor)
class TorchVector(Vector):
    """Vector subclass to store PyTorch tensors.

    This Vector is designed to store a collection of named
    PyTorch tensors, enabling computations that are either
    applied to each and every coefficient, or imply two sets
    of aligned coefficients (i.e. two TorchVector with
    similar specifications).
    """

    def __init__(
            self,
            coefs: Dict[str, torch.Tensor]
        ) -> None:
        super().__init__(coefs)

    def serialize(
            self,
        ) -> str:
        data = {
            key: serialize_numpy(tns.numpy())
            for key, tns in self.coefs.items()
        }
        return json.dumps(data)

    @classmethod
    def deserialize(
            cls,
            string: str,
        ) -> 'TorchVector':
        # false-positive; pylint: disable=no-member
        data = json.loads(string)
        coef = {
            key: torch.from_numpy(deserialize_numpy(dat).copy())
            for key, dat in data.items()
        }
        return cls(coef)

    def __eq__(
            self,
            other: Any,
        ) -> bool:
        valid = isinstance(other, TorchVector)
        valid &= (self.coefs.keys() == other.coefs.keys())
        return valid and all(
            np.array_equal(self.coefs[k].numpy(), other.coefs[k].numpy())
            for k in self.coefs
        )

    def _apply_operation(
            self,
            other: Any,
            func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ) -> 'TorchVector':
        """Apply an operation to combine this vector with another."""
        # Case when operating on two TorchVector objects.
        if isinstance(other, (TorchVector, NumpyVector)):
            if self.coefs.keys() != other.coefs.keys():
                raise KeyError(
                    f"Cannot {func.__name__} TorchVectors "\
                    "with distinct coefficient names."
                )
            return TorchVector({
                key: func(self.coefs[key], other.coefs[key])
                for key in self.coefs
            })
        # Case when operating with another object (e.g. a scalar).
        try:
            return TorchVector({
                key: func(coef, other)
                for key, coef in self.coefs.items()
            })
        except TypeError as exc:
            raise TypeError(
                f"Cannot {func.__name__} TorchVector "\
                f"with object of type {type(other)}."
            ) from exc

    def apply_func(
            self,
            func: Callable[[torch.Tensor], torch.Tensor],
            *args: Any,
            **kwargs: Any
        ) -> 'TorchVector':
        """Apply a tensor-altering function to the wrapped coefficients."""
        return TorchVector({
            key: func(coef, *args, **kwargs)
            for key, coef in self.coefs.items()
        })

    def __add__(
            self,
            other: Any
        ) -> 'TorchVector':
        # false-positive; pylint: disable=no-member
        return self._apply_operation(other, torch.add)

    def __sub__(
            self,
            other: Any
        ) -> 'TorchVector':
        # false-positive; pylint: disable=no-member
        return self._apply_operation(other, torch.sub)

    def __mul__(
            self,
            other: Any
        ) -> 'TorchVector':
        # false-positive; pylint: disable=no-member
        return self._apply_operation(other, torch.mul)

    def __truediv__(
            self,
            other: Any
        ) -> 'TorchVector':
        # false-positive; pylint: disable=no-member
        return self._apply_operation(other, torch.div)

    def __pow__(
            self,
            power: float,
            modulo: Optional[int] = None
        ) -> 'TorchVector':
        # false-positive; pylint: disable=no-member
        return self.apply_func(torch.pow, power)  # type: ignore

    def sign(
            self
        ) -> 'TorchVector':
        # false-positive; pylint: disable=no-member
        return self.apply_func(torch.sign)

    def minimum(
            self,
            other: Any,
        ) -> 'TorchVector':
        # false-positive; pylint: disable=no-member
        if isinstance(other, Vector):
            return self._apply_operation(other, torch.minimum)
        return self.apply_func(torch.minimum, other)  # type: ignore

    def maximum(
            self,
            other: Any,
        ) -> 'TorchVector':
        # false-positive; pylint: disable=no-member
        if isinstance(other, Vector):
            return self._apply_operation(other, torch.minimum)
        return self.apply_func(torch.maximum, other)  # type: ignore
