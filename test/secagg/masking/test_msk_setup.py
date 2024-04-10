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

from declearn.secagg.masking import (
    MaskingDecrypter,
    MaskingEncrypter,
    MaskingSecaggConfigClient,
    MaskingSecaggConfigServer,
)
from declearn.test_utils import MockNetworkClient, MockNetworkServer


@pytest.fixture(name="id_keys", scope="module")
def id_keys_fixture() -> List[Ed25519PrivateKey]:
    """Provide with random Ed25519 keys generated once for this test file."""
    return [Ed25519PrivateKey.generate() for _ in range(5)]


async def run_server_routine(
    n_clients: int,
    bitsize: int,
    clipval: float,
) -> MaskingDecrypter:
    """Prepare for and run the server-side setup routine."""
    config = MaskingSecaggConfigServer(bitsize=bitsize, clipval=clipval)
    async with MockNetworkServer() as netwk:
        await netwk.wait_for_clients(n_clients)
        decrypter = await config.setup_decrypter(netwk)
    return decrypter


async def run_client_routine(
    name: str,
    prv_key: Ed25519PrivateKey,
    trusted: List[Ed25519PublicKey],
) -> MaskingEncrypter:
    """Prepare for and run the client-side setup routine."""
    config = MaskingSecaggConfigClient.from_params(
        id_keys={"prv_key": prv_key, "trusted": trusted}
    )
    async with MockNetworkClient(name=name) as netwk:
        await netwk.register({})
        msg = await netwk.recv_message()
        encrypter = await config.setup_encrypter(netwk, msg)
    return encrypter


@pytest.mark.parametrize("n_clients", [2, 5])
@pytest.mark.asyncio
async def test_masking_secagg_setup(
    n_clients: int,
    id_keys: List[Ed25519PrivateKey],
) -> None:
    """Test that the Joye-Libert setup routines work properly."""
    # Use arbitrary, non-default values.
    bitsize = 32
    clipval = 100.0
    # Setup the server and client routines.
    trusted = [key.public_key() for key in id_keys[:n_clients]]
    client_routines = [
        run_client_routine(f"client_{i}", id_keys[i], trusted)
        for i in range(n_clients)
    ]
    server_routine = run_server_routine(n_clients, bitsize, clipval)
    # Run the routines concurrently and gather resulting objects.
    decrypter, *encrypters = await asyncio.gather(
        server_routine, *client_routines
    )
    # Verify that the decrypter has expected types and hyper-parameters.
    assert isinstance(decrypter, MaskingDecrypter)
    assert decrypter.quantizer.val_range == clipval
    assert decrypter.quantizer.int_range * n_clients < 2**bitsize
    # Verify that the encrypters have expected types and hyper-parameters.
    encrypted = []  # type: List[int]
    for enc in encrypters:
        assert isinstance(enc, MaskingEncrypter)
        assert enc.quantizer.val_range == clipval
        assert enc.quantizer.int_range == decrypter.quantizer.int_range
        encrypted.append(enc.encrypt_uint(1))
    # Verify that the setup encrypters and decrypter work properly.
    assert all(x != 1 for x in encrypted)
    assert decrypter.decrypt_uint(sum(encrypted)) == n_clients
