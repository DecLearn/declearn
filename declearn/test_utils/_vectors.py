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

"""Shared objects for testing purposes."""

import importlib
import importlib.metadata
import typing
from typing import List, Literal, Optional, Type

import numpy as np
from numpy.typing import ArrayLike

from declearn.model.api import Vector
from declearn.model.sklearn import NumpyVector

__all__ = [
    "FrameworkType",
    "GradientsTestCase",
    "list_available_frameworks",
]


FrameworkType = Literal["numpy", "tensorflow", "torch", "jax"]


def list_available_frameworks() -> List[FrameworkType]:
    """List available Vector backend frameworks."""
    available = []
    for framework in typing.get_args(FrameworkType):
        try:
            importlib.metadata.distribution(framework)
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            available.append(framework)
    return available


class GradientsTestCase:
    """Framework-parametrized Vector instances provider for testing purposes.

    This class aims at providing with seeded random or zero-valued Vector
    instances (with deterministic specifications) that may be used in the
    context of unit tests.
    """

    def __init__(
        self,
        framework: FrameworkType,
        seed: Optional[int] = 0,
    ) -> None:
        """Instantiate the parametrized test-case."""
        if framework not in list_available_frameworks():
            raise RuntimeError(f"Framework '{framework}' is unavailable.")
        self.framework = framework
        self.seed = seed

    @property
    def vector_cls(self) -> Type[Vector]:
        """Vector subclass suitable to the tested framework."""
        if self.framework == "numpy":
            return NumpyVector
        if self.framework == "tensorflow":
            module = importlib.import_module("declearn.model.tensorflow")
            return module.TensorflowVector
        if self.framework == "torch":
            module = importlib.import_module("declearn.model.torch")
            return module.TorchVector
        if self.framework == "jax":
            module = importlib.import_module("declearn.model.haiku")
            return module.JaxNumpyVector
        raise ValueError(f"Invalid framework '{self.framework}'")

    def convert(self, array: np.ndarray) -> ArrayLike:
        """Convert an input numpy array to a framework-based structure."""
        if self.framework == "numpy":
            return array
        if self.framework == "tensorflow":
            tensorflow = importlib.import_module("tensorflow")
            with tensorflow.device("CPU"):
                return tensorflow.convert_to_tensor(array)
        if self.framework == "torch":
            torch = importlib.import_module("torch")
            return torch.from_numpy(array)
        if self.framework == "jax":
            jnp = importlib.import_module("jax.numpy")
            return jnp.asarray(array)
        raise ValueError(f"Invalid framework '{self.framework}'")

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
        vector = self.vector_cls(
            {str(idx): self.convert(value) for idx, value in enumerate(values)}
        )
        # In Tensorflow, convert the first gradients to IndexedSlices.
        # In this case they are equivalent to dense ones, but this enables
        # testing the support for these structures while maintaining the
        # possibility to compare outputs' values with other frameworks.
        if self.framework == "tensorflow":
            tensorflow = importlib.import_module("tensorflow")
            vector.coefs["0"] = tensorflow.IndexedSlices(
                values=vector.coefs["0"],
                indices=tensorflow.range(64),
                dense_shape=tensorflow.convert_to_tensor([64, 32]),
            )
        return vector

    @property
    def mock_ones(self) -> Vector:
        """Instantiate a Vector with random-valued mock gradients.

        Note: the RNG used to generate gradients has a fixed seed,
                to that gradients have the same values whatever the
                tensor framework used is.
        """
        shapes = [(5, 5), (4,), (1,)]
        values = [np.ones(shape) for shape in shapes]
        return self.vector_cls(
            {str(idx): self.convert(value) for idx, value in enumerate(values)}
        )

    @property
    def mock_zeros(self) -> Vector:
        """Instantiate a Vector with random-valued mock gradients.

        Note: the RNG used to generate gradients has a fixed seed,
                to that gradients have the same values whatever the
                tensor framework used is.
        """
        shapes = [(5, 5), (4,), (1,)]
        values = [np.zeros(shape) for shape in shapes]
        return self.vector_cls(
            {str(idx): self.convert(value) for idx, value in enumerate(values)}
        )
