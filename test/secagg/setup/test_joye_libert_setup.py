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

"""Unit tests for Joye-Libert setup routines."""

import asyncio
import os
from typing import Any, Dict, Tuple
from unittest import mock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from declearn.secagg.joye_libert import (
    JoyeLibertDecrypter,
    JoyeLibertEncrypter,
    JoyeLibertSecaggConfigClient,
    JoyeLibertSecaggConfigServer,
)
from declearn.secagg.joye_libert.messages import JoyeLibertSecaggSetupQuery
from declearn.secagg.utils import generate_random_biprime
from declearn.test_utils import make_importable

with make_importable(os.path.dirname(os.path.abspath(__file__))):
    from secagg_setup_testing import SecaggSetupTestCase


class TestJoyeLibertSecaggSetup(SecaggSetupTestCase):
    """Unit tests for Joye-Libert SecAgg setup and config classes."""

    decrypter_cls = JoyeLibertDecrypter
    encrypter_cls = JoyeLibertEncrypter
    client_config_cls = JoyeLibertSecaggConfigClient
    server_config_cls = JoyeLibertSecaggConfigServer
    setup_msg_cls = JoyeLibertSecaggSetupQuery

    def get_client_hyper_parameters(
        self,
    ) -> Dict[str, Any]:
        biprime = generate_random_biprime(half_bitsize=32)
        return {"biprime": biprime}

    def assert_decrypter_validity(
        self,
        decrypter: JoyeLibertDecrypter,  # type: ignore[override]
        **kwargs: Any,
    ) -> None:
        # Verify decrypter type and quantization clipval.
        super().assert_decrypter_validity(decrypter, **kwargs)
        # Verify biprime value and quantization integer range.
        assert decrypter.biprime == kwargs["biprime"]
        assert decrypter.quantizer.int_range == 2 ** kwargs["bitsize"] - 1

    def assert_encrypter_validity(
        self,
        encrypter: JoyeLibertEncrypter,  # type: ignore[override]
        decrypter: JoyeLibertDecrypter,  # type: ignore[override]
        **kwargs: Any,
    ) -> None:
        # Verify encrypter type and quantization parameters coherence.
        super().assert_encrypter_validity(encrypter, decrypter, **kwargs)
        # Verify biprime value and private key bitsize.
        biprime = decrypter.biprime
        assert encrypter.biprime == biprime
        assert encrypter.prv_key.bit_length() <= 2 * biprime.bit_length()

    @pytest.mark.asyncio
    async def test_joye_libert_secagg_setup_disparate_biprime(
        self,
    ) -> None:
        """Test that the setup fails if clients specify distinct biprimes."""
        # Generate identity keys.
        id_keys = [Ed25519PrivateKey.generate() for _ in range(2)]
        trusted = [key.public_key() for key in id_keys]
        # Set up routines where the clients use distinct biprime values.
        kwargs = self.get_server_hyper_parameters()
        server_routine = self.run_server_routine(n_clients=2, **kwargs)
        peer_0_routine = self.run_client_routine(
            "0", id_keys[0], trusted, biprime=generate_random_biprime(8)
        )
        peer_1_routine = self.run_client_routine(
            "1", id_keys[1], trusted, biprime=generate_random_biprime(16)
        )
        # Test that as a result, the setup fails.
        result: Tuple[
            RuntimeError, RuntimeError, RuntimeError
        ] = await asyncio.gather(
            server_routine,
            peer_0_routine,
            peer_1_routine,
            return_exceptions=True,
        )
        server_exc, peer_0_exc, peer_1_exc = result
        assert isinstance(server_exc, RuntimeError)
        assert isinstance(peer_0_exc, RuntimeError)
        assert isinstance(peer_1_exc, RuntimeError)

    @pytest.mark.asyncio
    async def test_joye_libert_secagg_setup_fake_shamir_prime(
        self,
    ) -> None:
        """Test that the setup fails if server emits a fake prime number."""
        # Generate identity keys and a common biprime number.
        id_keys = [Ed25519PrivateKey.generate() for _ in range(2)]
        trusted = [key.public_key() for key in id_keys]
        biprime = generate_random_biprime(16)

        # Define a malicious server routine.
        async def malicious_server_routine(
            n_clients: int,
            **kwargs: Any,
        ) -> JoyeLibertDecrypter:
            """Patch the server routine to send a non-prime number."""
            with mock.patch(
                "declearn.secagg.joye_libert._setup.generate_random_prime",
                return_value=13839048920,  # proper bitsize, but not a prime
            ):
                decrypter = await self.run_server_routine(n_clients, **kwargs)
                return decrypter  # type: ignore[return-value]

        # Set up routines where the clients use distinct biprime values.
        kwargs = self.get_server_hyper_parameters()
        server_routine = malicious_server_routine(n_clients=2, **kwargs)
        peer_0_routine = self.run_client_routine(
            "0", id_keys[0], trusted, biprime=biprime
        )
        peer_1_routine = self.run_client_routine(
            "0", id_keys[1], trusted, biprime=biprime
        )
        # Test that as a result, the setup fails.
        result: Tuple[
            RuntimeError, RuntimeError, RuntimeError
        ] = await asyncio.gather(
            server_routine,
            peer_0_routine,
            peer_1_routine,
            return_exceptions=True,
        )
        server_exc, peer_0_exc, peer_1_exc = result
        assert isinstance(server_exc, RuntimeError)
        assert isinstance(peer_0_exc, RuntimeError)
        assert isinstance(peer_1_exc, RuntimeError)
