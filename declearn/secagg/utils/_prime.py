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

"""Random prime number generation utils."""

import secrets

import gmpy2  # type: ignore

__all__ = [
    "generate_random_biprime",
    "generate_random_prime",
]


def generate_random_biprime(
    half_bitsize: int = 1024,
) -> int:
    """Generate a random biprime integer with a target bit length.

    Parameters
    ----------
    half_bitsize:
        Bit length of the two random prime integers that make up
        the output biprime number.

    Returns
    -------
    biprime:
        Integer obtained by multiplying two prime numbers of equal
        bit length `half_bitsize`, the bit length of which is thus
        either `2 * half_bitsize` or `2 * half_bitsize - 1`.
    """
    n_prime = generate_random_prime(half_bitsize)
    q_prime = generate_random_prime(half_bitsize)
    return int(gmpy2.mul(n_prime, q_prime))


def generate_random_prime(
    bitsize: int,
) -> int:
    """Generate a random prime integer with given bit length.

    Parameters
    ----------
    bitsize:
        Bit length of the returned prime number.

    Returns
    -------
    prime:
        Prime integer in [2**(bitzize-1), 2**bitsize(.
    """
    value = secrets.randbits(bitsize)
    while not gmpy2.is_prime(value):
        value += 1
    if value.bit_length() != bitsize:  # pragma: no cover
        return generate_random_prime(bitsize)
    return int(value)
