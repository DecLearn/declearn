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

"""Functional tests for NoiseModule subclasses.

* Test that a given seed returns the same thing twice.
* Test that random generation returns different vectors each time.
* Test that the correct gaussian noise is added to the gradient.
"""

from typing import Type

import pytest
from scipy import stats  # type: ignore

from declearn.optimizer.modules import GaussianNoiseModule, NoiseModule
from declearn.test_utils import FrameworkType, GradientsTestCase


NOISETYPES = NoiseModule.__subclasses__()


@pytest.mark.parametrize("cls", NOISETYPES)
class TestNoiseModule:
    """Functional tests for declearn.optimizer.modules.NoiseModule subclasses.

    This class implements a series of generic tests that focus
    on the RNG's proper behavior and are therefore suitable for
    all NoiseModule subclasses.
    """

    @pytest.mark.parametrize("seed", [0, 123])
    def test_seed_reproducibility(
        self, cls: Type[NoiseModule], framework: FrameworkType, seed: int
    ) -> None:
        """Test that using the same seed twice returns the same vector."""
        grad = GradientsTestCase(framework=framework, seed=seed).mock_gradient
        noisy_grads = []
        for _ in range(2):
            noise_module = cls(safe_mode=False, seed=seed)
            noisy_grads.append(noise_module.run(grad))
        assert noisy_grads[0] == noisy_grads[1]

    @pytest.mark.parametrize("safe_mode", [False, True])
    def test_randomness(
        self, cls: Type[NoiseModule], framework: FrameworkType, safe_mode: bool
    ) -> None:
        """Test that different random calls return different results."""
        grad = GradientsTestCase(framework=framework).mock_gradient
        noisy_grads = []
        for _ in range(2):
            noise_module = cls(safe_mode=safe_mode)
            noisy_grads.append(noise_module.run(grad))
        assert noisy_grads[0] != noisy_grads[1]


@pytest.mark.parametrize("safe_mode", [False, True])
class TestGaussianNoiseModule:
    """Functional tests for declearn.optimizer.modules.GaussianNoiseModule.

    Note this these tests are only run using NumpyVector inputs.
    """

    # pylint: disable=too-few-public-methods

    @pytest.mark.parametrize("std", [1.0, 10e-5, 100.0])
    def test_distribution(self, safe_mode: bool, std: int) -> None:
        """Test that the noise's average and stdev are statistically correct.

        Assesses goodness of fit using the two-sided Kolmogorov-Smirnov test
        with a confidence level of 0.995.

        For more details, see :
        https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test

        Note: this test aggregates the noise added on all elements of the
        gradient, since they should all come from the same distribution.
        """
        grad = GradientsTestCase(framework="numpy").mock_gradient
        module = GaussianNoiseModule(std=std, safe_mode=safe_mode)
        noisy_grad = module.run(grad)
        just_noise = noisy_grad - grad
        noise_list = [  # flattened list of noise values
            value
            for coef in just_noise.coefs.values()
            for value in coef.flatten().tolist()
        ]
        assert stats.kstest(noise_list, "norm", args=(0, std))[1] > 0.005
