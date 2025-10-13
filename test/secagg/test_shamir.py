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

"""Unit tests for Shamir Secret Sharing tools."""

import itertools
import secrets
from typing import Optional

import numpy as np
import pytest

from declearn.secagg.shamir import (
    generate_secret_shares,
    recover_shared_secret,
)


def test_generate_secret_shares_raises_on_invalid_secret():
    """Test that 'generate_secret_shares' raises on invalid 'secret' input."""
    with pytest.raises(TypeError):
        generate_secret_shares(secret=1234.0, shares=2)


def test_generate_secret_shares_raises_on_invalid_shares():
    """Test that 'generate_secret_shares' raises on invalid 'shares' input."""
    with pytest.raises(TypeError):
        generate_secret_shares(secret=1234, shares=-1)


def test_generate_secret_shares_raises_on_invalid_thresh():
    """Test that 'generate_secret_shares' raises on invalid 'thresh' input."""
    with pytest.raises(TypeError):
        generate_secret_shares(secret=1234, shares=2, thresh=4)


def test_generate_secret_shares_raises_on_invalid_xcoord():
    """Test that 'generate_secret_shares' raises on invalid 'xcoord' input."""
    # Test with the wrong type of input.
    with pytest.raises(TypeError):
        generate_secret_shares(secret=1234, shares=2, xcoord="wrong-type")
    # Test with too many values.
    with pytest.raises(TypeError):
        generate_secret_shares(secret=1234, shares=2, xcoord=[1, 2, 3])
    # Test with duplicate values.
    with pytest.raises(TypeError):
        generate_secret_shares(secret=1234, shares=2, xcoord=[1, 1])
    # Test with a zero-valued input.
    with pytest.raises(TypeError):
        generate_secret_shares(secret=1234, shares=2, xcoord=[0, 1])
    # Test with a negative value.
    with pytest.raises(TypeError):
        generate_secret_shares(secret=1234, shares=2, xcoord=[-2, 4])


def test_generate_secret_shares_raises_on_invalid_mprime():
    """Test that 'generate_secret_shares' raises on invalid 'mprime' input."""
    # Test with a non-prime number.
    with pytest.raises(TypeError):
        generate_secret_shares(secret=1234, shares=2, mprime=42)
    # Test with a small prime number.
    with pytest.raises(TypeError):
        generate_secret_shares(secret=1234, shares=2, mprime=17)


@pytest.mark.parametrize("x_rand", [False, True], ids=["xdefault", "x_random"])
@pytest.mark.parametrize(["shares", "thresh"], [(4, None), (4, 2), (12, 9)])
def test_secret_sharing(
    shares: int,
    thresh: Optional[int],
    x_rand: bool,
) -> None:
    """Test that Shamir secret sharing tools work properly.

    Verify that for given (k, n) parameters and a random secret integer,
    any combination of >=k shares enables recovering the secret, while
    any combination of <k shares results in a wrong reconstructed value.
    """
    if x_rand:
        xcoord = (
            1 + np.random.choice(2**8, size=shares, replace=False)
        ).tolist()
    else:
        xcoord = None
    # Generate a secret value and shares to it.
    secret = secrets.randbits(32)
    shamir = generate_secret_shares(
        secret, shares=shares, thresh=thresh, xcoord=xcoord
    )
    if thresh is None:
        thresh = shares
    # Verify that the outputs have proper format.
    assert isinstance(shamir, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in shamir)
    assert [idx for idx, _ in shamir] == (xcoord or list(range(1, shares + 1)))
    assert all(isinstance(val, int) for _, val in shamir)
    # Verify that the secret can be recovered by any combination of k+ shares.
    assert all(
        secret == recover_shared_secret(shares=[shamir[i] for i in idx])
        for n in range(thresh, shares + 1)
        for idx in itertools.combinations(range(shares), n)
    )
    # Verify that the secret cannot be recovered from any subset of <k shares.
    assert all(
        secret != recover_shared_secret(shares=[shamir[i] for i in idx])
        for n in range(1, thresh)
        for idx in itertools.combinations(range(shares), n)
    )
