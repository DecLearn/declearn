# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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

"""Unit tests for random prime number generation utils."""

import gmpy2  # type: ignore
import pytest

from declearn.secagg.utils import (
    generate_random_biprime,
    generate_random_prime,
)


@pytest.mark.parametrize("bitsize", [8, 128, 1024])
def test_generate_random_prime(bitsize: int) -> None:
    """Test that `generate_random_prime` works properly."""
    primes = [generate_random_prime(bitsize) for _ in range(10)]
    assert all(isinstance(x, int) for x in primes)
    assert all(x.bit_length() == bitsize for x in primes)
    assert all(gmpy2.is_prime(x) for x in primes)
    assert len(set(primes)) > 1


@pytest.mark.parametrize("half_bitsize", [8, 128, 1024])
def test_generate_random_biprime(half_bitsize: int) -> None:
    """Test that `generate_random_biprime` works properly."""
    biprime = generate_random_biprime(half_bitsize)
    assert isinstance(biprime, int)
    assert (2 * half_bitsize - 1) <= biprime.bit_length() <= (2 * half_bitsize)
