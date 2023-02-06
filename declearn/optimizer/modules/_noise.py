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

"""Noise-addition modules for DP using cryptographically-strong RNG."""

from abc import ABCMeta, abstractmethod
from random import SystemRandom
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import scipy.stats  # type: ignore

from declearn.model.api import Vector
from declearn.model.sklearn import NumpyVector
from declearn.optimizer.modules._api import OptiModule

__all__ = [
    "GaussianNoiseModule",
    "NoiseModule",
]


class NoiseModule(OptiModule, metaclass=ABCMeta, register=False):
    """Abstract noise-addition module for DP purposes.

    This module uses either fast numpy pseudo-random number generation,
    or slower cryptographically secure pseudo-random numbers (CSPRN).
    """

    name: ClassVar[str] = "abstract-noise"

    def __init__(
        self,
        safe_mode: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Instantiate the noise module.

        Parameters
        ----------
        safe_mode: bool, default=True
            Whether to use cryptographically-secure pseudo-random numbers
            (CSPRN) rather than the default numpy generator.
            For experimental purposes, set flag to False, as generating CSPRN
            is significantly slower.
        seed: int or None, default=None
            Seed used for initiliazing the non-secure random number generator.
            If `safe_mode=True`, seed is ignored.
        """
        rng = SystemRandom if safe_mode else np.random.default_rng
        self._rng = rng(seed)
        self.seed = seed

    @property
    def safe_mode(self) -> bool:
        """Whether this module uses CSPRN rather than base numpy RNG."""
        return isinstance(self._rng, SystemRandom)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        return {"safe_mode": self.safe_mode, "seed": self.seed}

    def run(
        self,
        gradients: Vector,
    ) -> Vector:
        if not NumpyVector in gradients.compatible_vector_types:
            raise TypeError(
                f"{self.__class__.__name__} requires input gradients to "
                "be compatible with NumpyVector, which is not the case "
                f"of {type(gradients).__name__}."
            )
        # Gather gradients' specs.
        shapes = gradients.shapes()
        dtypes = gradients.dtypes()
        # Conduct noise sampling for each and every gradient coordinate.
        noise = {
            key: self._sample_noise(shapes[key], dtypes[key])
            for key in gradients.coefs
        }
        # Add the sampled noise to the gradients and return them.
        return gradients + NumpyVector(noise)

    @abstractmethod
    def _sample_noise(
        self,
        shape: Tuple[int, ...],
        dtype: str,
    ) -> np.ndarray:
        """Sample a noise tensor from a module-specific distribution."""


class GaussianNoiseModule(NoiseModule):
    """Gaussian-noise addition module for DP-SGD.

    This module uses either fast numpy pseudo-random number generation,
    or slower cryptographically secure pseudo-random numbers (CSPRN).
    """

    name: ClassVar[str] = "gaussian-noise"

    def __init__(
        self,
        std: float = 1.0,
        safe_mode: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Instantiate the gaussian noise module.

        Parameters
        ----------
        std: float, default=1.0,
            Standard deviation of the gaussian noise.
        safe_mode: bool, default=True
            Whether to use cryptographically-secure pseudo-random numbers
            (CSPRN) rather than the default numpy generator.
            For experimental purposes, set flag to False, as generating CSPRN
            is significantly slower.
        seed: int or None, default=None
            Seed used for initiliazing the non-secure random number generator.
            If `safe_mode=True`, seed is ignored.
        """
        self.std = std
        super().__init__(safe_mode, seed)

    def get_config(
        self,
    ) -> Dict[str, Any]:
        config = super().get_config()
        config["std"] = self.std
        return config

    def _sample_noise(
        self,
        shape: Tuple[int, ...],
        dtype: str,
    ) -> np.ndarray:
        """Sample a noise tensor from a Gaussian distribution."""
        # Case when using CSPRN, that only provides uniform value sampling.
        # REVISE: improve performance, possibly using torch's CSPRN lib
        if isinstance(self._rng, SystemRandom):
            value = [self._rng.random() for _ in range(np.prod(shape))]
            array = np.array(value).reshape(shape).astype(dtype)
            return scipy.stats.norm.ppf(array, scale=self.std)
        # Case when using numpy RNG, that provides with gaussian sampling.
        if isinstance(self._rng, np.random.Generator):
            # false-positive; pylint: disable=no-member
            return self._rng.normal(scale=self.std, size=shape).astype(dtype)
        # Theoretically-unreachable case.
        raise RuntimeError("Unexpected `GaussianeNoiseModule._rng` type.")
