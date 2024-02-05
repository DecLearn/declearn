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

"""Unit tests for Joye-Libert setup routines."""

import asyncio
from typing import List

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from declearn.secagg.joye_libert import (
    JoyeLibertDecrypter,
    JoyeLibertEncrypter,
)
from declearn.secagg.setup import ClientJoyeLibertSetup, ServerJoyeLibertSetup
from declearn.secagg.utils import generate_random_biprime
from declearn.test_utils import MockNetworkClient, MockNetworkServer


@pytest.fixture(name="id_keys", scope="module")
def id_keys_fixture() -> List[Ed25519PrivateKey]:
    """Provide with random Ed25519 keys generated once for this test file."""
    return [Ed25519PrivateKey.generate() for _ in range(5)]


async def run_server_routine(
    n_clients: int,
    bitsize: int,
    clipval: float,
) -> JoyeLibertDecrypter:
    """Prepare for and run the server-side setup routine."""
    async with MockNetworkServer() as netwk:
        await netwk.wait_for_clients(n_clients)
        routine = ServerJoyeLibertSetup(
            netwk, bitsize=bitsize, clipval=clipval
        )
        decrypter = await routine.async_run()
    return decrypter


async def run_client_routine(
    name: str,
    prv_key: Ed25519PrivateKey,
    trusted: List[Ed25519PublicKey],
    biprime: int,
) -> JoyeLibertEncrypter:
    """Prepare for and run the client-side setup routine."""
    async with MockNetworkClient(name=name) as netwk:
        await netwk.register({})
        routine = ClientJoyeLibertSetup(
            netwk, prv_key=prv_key, trusted=trusted, biprime=biprime
        )
        encrypter = await routine.async_run()
    return encrypter


@pytest.mark.parametrize("n_clients", [2, 5])
@pytest.mark.asyncio
async def test_joye_libert_setup_routines(
    n_clients: int,
    id_keys: List[Ed25519PrivateKey],
) -> None:
    """Test that the Joye-Libert setup routines work properly."""
    # Use arbitrary, non-default values.
    bitsize = 16
    clipval = 100.0
    biprime = generate_random_biprime(half_bitsize=32)
    # Setup the server and client routines.
    trusted = [key.public_key() for key in id_keys[:n_clients]]
    client_routines = [
        run_client_routine(f"client_{i}", id_keys[i], trusted, biprime)
        for i in range(n_clients)
    ]
    server_routine = run_server_routine(n_clients, bitsize, clipval)
    # Run the routines concurrently and gather resulting objects.
    decrypter, *encrypters = await asyncio.gather(
        server_routine, *client_routines
    )
    # Verify that the decrypter has expected types and hyper-parameters.
    assert isinstance(decrypter, JoyeLibertDecrypter)
    assert decrypter.biprime == biprime
    assert decrypter.quantizer.val_range == clipval
    assert decrypter.quantizer.int_range == 2**bitsize - 1
    # Verify that the encrypters have expected types and hyper-parameters.
    prv_keys = []  # type: List[int]
    for enc in encrypters:
        assert isinstance(enc, JoyeLibertEncrypter)
        assert enc.biprime == biprime
        assert enc.quantizer.val_range == clipval
        assert enc.quantizer.int_range == 2**bitsize - 1
        prv_keys.append(enc.prv_key)
    # Verify that the public key matches the private ones.
    assert decrypter.pub_key == -sum(prv_keys)
    # Verify that private keys have the expected bit size.
    exp_size = 2 * biprime.bit_length()
    assert all(key.bit_length() <= exp_size for key in prv_keys)
