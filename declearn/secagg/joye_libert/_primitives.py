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

"""Joye-Libert Homomorphic Summation tools."""

import hashlib
from typing import List, Tuple, Union

import gmpy2  # type: ignore

__all__ = [
    "DEFAULT_BIPRIME",
    "encrypt",
    "decrypt_sum",
    "sum_encrypted",
]


# Default biprime number, with 1023 bitsize.
PRIME_P = int(
    "7801876574383880214548650574033350741129913580793719"
    "7067463616060425410801412911322248991130479347607911"
    "08387050756752894517232516965892712015132079112571"
)
PRIME_Q = int(
    "7755946847853454424709929267431997195175500554762787"
    "7152471113855966527410223993208656880021149734530570"
    "88521173384791077635017567166681500095602864712097"
)
DEFAULT_BIPRIME = int(gmpy2.mul(PRIME_P, PRIME_Q))
"""Default Biprime value used as modulus in Joye-Libert functions."""


def hash_into_domain(
    index: int,
    modulus: int,
) -> int:
    """Hash a positive integer value into the (Z/Zp²Z)* domain.

    Parameters
    ----------
    index:
        Positive integer value to hash.
    modulus:
        Modulus with respect to which to define the subdomain
        `{x in {0, ..., mod**2} | gcd(x, mod) = 1}`.

    Returns
    -------
    hash:
        Integer value resulting from hashing `index` onto the target domain.
    """
    # Convert index to bytes and initialize some values.
    m_square = gmpy2.square(modulus)
    bitsize = modulus.bit_length()
    b_indx = index.to_bytes(bitsize // 2, "big")
    result = gmpy2.mpz(0)
    counter = 0
    # Iterate until a hash belonging to the domain is found.
    while gmpy2.gcd(result, modulus) != 1:
        rbytes = b""
        while len(rbytes) < max(bitsize // 8, 1):
            rbytes += hashlib.sha256(
                b_indx + counter.to_bytes(1, "big")
            ).digest()
            counter += 1
        result = gmpy2.mpz(int.from_bytes(rbytes[-bitsize:], "big")) % m_square
    # Return the resulting value.
    return int(result)


def encrypt(
    value: int,
    index: int,
    secret: int,
    modulus: int = DEFAULT_BIPRIME,
) -> int:
    """Apply Joye-Libert encryption to a positive integer value.

    Parameters
    ----------
    value:
        Private positive integer that needs encrypting.
    index:
        Public encryption index.
    secret:
        Private key (large integer) used to encrypt the value.
    modulus:
        Public biprime modulus value.

    Returns
    -------
    crypted:
        Crypted transform of the private value.
        Positive int in `[0, modulus**2[`.
    """
    m_square = gmpy2.square(modulus)
    h_t = hash_into_domain(index, modulus)
    hpm = gmpy2.powmod(h_t, secret, m_square)
    return int(((1 + value * modulus) * hpm) % m_square)


def sum_encrypted(
    values: Union[Tuple[int, ...], List[int]],
    modulus: int = DEFAULT_BIPRIME,
) -> int:
    """Apply homomorphic summation to some Joye-Libert encrypted values.

    Parameters
    ----------
    values:
        List (or tuple) of encrypted private values that need summation.
    modulus:
        Public biprime modulus value, defining the integer field to
        which encrypted values belong.

    Returns
    -------
    sum_of_values:
        Decrypted sum of the private values.

    Raises
    ------
    TypeError
        If `values` is an empty list.
    """
    if not values:
        raise TypeError("Cannot sum an empty list of encrypted values.")
    m_square = gmpy2.square(modulus)
    output = gmpy2.mpz(values[0])
    for value in values[1:]:
        output = (output * value) % m_square
    return int(output)


def decrypt_sum(
    sum_of_values: int,
    index: int,
    public: int,
    modulus: int = DEFAULT_BIPRIME,
) -> int:
    """Apply Joye-Libert decryption to an encrypted sum of private values.

    Parameters
    ----------
    sum_of_values:
        Encrypted value resulting from the aggregation of private ones,
        using the `sum_encrypted` operator (homomorphic to summation).
    index:
        Public encryption index.
    public:
        Public negative sum of private integer keys used to encrypt `values`.
    modulus:
        Public biprime modulus value.

    Returns
    -------
    sum_of_values:
        Decrypted public sum of the encrypted private values.
    """
    m_square = gmpy2.square(modulus)
    h_t = hash_into_domain(index, modulus)
    v_t = gmpy2.powmod(h_t, public, m_square)
    v_t = (v_t * sum_of_values) % m_square
    return int((v_t - 1) // modulus)
