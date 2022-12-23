# coding: utf-8

"""Shared objects for testing purposes."""

import typing
import warnings
from typing import Dict, Optional, Tuple, Type

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

with warnings.catch_warnings():  # silence tensorflow import-time warnings
    warnings.simplefilter("ignore")
    import tensorflow as tf  # type: ignore

import torch
from declearn.model.api import Vector
from declearn.model.sklearn import NumpyVector
from declearn.model.tensorflow import TensorflowVector
from declearn.model.torch import TorchVector


__all__ = [
    "FrameworkType",
    "Frameworks",
    "GradientsTestCase",
]


FrameworkType = Literal["numpy", "tflow", "torch"]
Frameworks = typing.get_args(FrameworkType)  # type: Tuple[FrameworkType, ...]


class GradientsTestCase:
    """Framework-parametrized Vector instances provider for testing purposes.

    This class aims at providing with seeded random or zero-valued Vector
    instances (with deterministic specifications) that may be used in the
    context of unit tests.
    """

    def __init__(
        self, framework: FrameworkType, seed: Optional[int] = 0
    ) -> None:
        """Instantiate the parametrized test-case."""
        self.framework = framework
        self.seed = seed

    @property
    def vector_cls(self) -> Type[Vector]:
        """Vector subclass suitable to the tested framework."""
        classes = {
            "numpy": NumpyVector,
            "tflow": TensorflowVector,
            "torch": TorchVector,
        }  # type: Dict[str, Type[Vector]]
        return classes[self.framework]

    def convert(self, array: np.ndarray) -> ArrayLike:
        """Convert an input numpy array to a framework-based structure."""
        functions = {
            "numpy": np.array,
            "tflow": tf.convert_to_tensor,
            "torch": torch.from_numpy,  # pylint: disable=no-member
        }
        return functions[self.framework](array)  # type: ignore

    def to_numpy(self, array: ArrayLike) -> np.ndarray:
        """Convert an input framework-based structure to a numpy array."""
        if isinstance(array, np.ndarray):
            return array
        return array.numpy()  # type: ignore

    @property
    def mock_gradient(self) -> Vector:
        """Instantiate a Vector with random-valued mock gradients.

        Note: the RNG used to generate gradients has a fixed seed,
              to that gradients have the same values whatever the
              tensor framework used is.
        """
        rng = np.random.default_rng(self.seed)
        shapes = [(64, 32), (32,), (32, 16), (16,), (16, 1), (1,)]
        values = [rng.normal(size=shape) for shape in shapes]
        return self.vector_cls(
            {str(idx): self.convert(value) for idx, value in enumerate(values)}
        )

    @property
    def mock_allzero_gradient(self) -> Vector:
        """Instantiate a Vector with random-valued mock gradients.

        Note: the RNG used to generate gradients has a fixed seed,
                to that gradients have the same values whatever the
                tensor framework used is.
        """
        shapes = [(64, 32), (32,), (32, 16), (16,), (16, 1), (1,)]
        values = [np.zeros(shape) for shape in shapes]
        return self.vector_cls(
            {str(idx): self.convert(value) for idx, value in enumerate(values)}
        )
