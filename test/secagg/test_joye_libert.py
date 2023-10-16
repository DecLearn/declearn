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

"""Unit tests for Joye-Libert homomorphic summation tools."""

import secrets

import pytest

from declearn.secagg.joye_libert import BIPRIME, encrypt, sum_decrypt


def test_encrypt_joye_libert_single_value() -> None:
    """Test that Joye-Libert encryption returns an int of expected size."""
    value = secrets.randbelow(2**256)
    p_key = secrets.randbelow(2**1024)
    c_val = encrypt(value, index=0, secret=p_key)
    assert isinstance(c_val, int)
    assert c_val.bit_length() <= (BIPRIME**2).bit_length()


def test_encrypt_joye_libert_variable_indices() -> None:
    """Test that Joye-Libert encryption is sensitive to the index choice."""
    value = secrets.randbelow(2**256)
    p_key = secrets.randbelow(2**1024)
    c_idx = [encrypt(value, index=i, secret=p_key) for i in range(32)]
    assert len(set(c_idx)) == 32


def test_encrypt_joye_libert_variable_keys() -> None:
    """Test that Joye-Libert encryption is sensitive to the key choice."""
    value = secrets.randbelow(2**32)
    index = secrets.randbelow(2**32) + 1
    c_idx = [
        encrypt(value, index, secret=secrets.randbelow(2**256))
        for _ in range(32)
    ]
    assert len(set(c_idx)) == 32


@pytest.mark.parametrize("index", [0, 1, 2**64])
def test_encrypt_decrypt_joye_libert(index: int) -> None:
    """Test that Joye-Libert encryption and sum-decryption works properly."""
    values = [secrets.randbelow(2**32) for _ in range(4)]
    p_keys = [secrets.randbelow(2**256) for _ in range(4)]
    c_vals = [
        encrypt(value, index=index, secret=secret)
        for value, secret in zip(values, p_keys)
    ]
    public = -sum(p_keys)
    result = sum_decrypt(c_vals, index, public)
    assert isinstance(result, int)
    assert result == sum(values)


def test_encrypt_decrypt_joye_libert_wrong_index() -> None:
    """Test that Joye-Libert sum-decryption fails with wrong index."""
    values = [secrets.randbelow(2**32) for _ in range(4)]
    p_keys = [secrets.randbelow(2**256) for _ in range(4)]
    index = secrets.randbelow(2**32) + 1
    c_vals = [
        encrypt(value, index=index, secret=secret)
        for value, secret in zip(values, p_keys)
    ]
    public = -sum(p_keys)
    result = sum_decrypt(c_vals, index + 1, public)
    assert isinstance(result, int)
    assert result != sum(values)


def test_encrypt_decrypt_joye_libert_wrong_public_key() -> None:
    """Test that Joye-Libert sum-decryption fails with wrong public key."""
    values = [secrets.randbelow(2**32) for _ in range(4)]
    p_keys = [secrets.randbelow(2**256) for _ in range(4)]
    index = secrets.randbelow(2**32) + 1
    c_vals = [
        encrypt(value, index=index, secret=secret)
        for value, secret in zip(values, p_keys)
    ]
    public = -sum(p_keys)
    result = sum_decrypt(c_vals, index, public + 1)
    assert isinstance(result, int)
    assert result != sum(values)
