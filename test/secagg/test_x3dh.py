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

"""Unit tests for the X3DH key agreement protocol implementation."""

import secrets
from typing import Tuple

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from declearn.secagg.x3dh import X3DHManager


@pytest.fixture(name="private_keys", scope="module")
def fixture_private_keys() -> Tuple[Ed25519PrivateKey, Ed25519PrivateKey]:
    """Fixture providing with a pair of Ed25519 private keys."""
    key_0 = Ed25519PrivateKey.generate()
    key_1 = Ed25519PrivateKey.generate()
    return key_0, key_1


class TestX3DHManager:
    """Unit tests for the X3DH key agreement protocol implementation."""

    # Tests for instantiation and request creation.

    def test_init(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test that 'X3DHManager' instantiation works properly."""
        # Setup a X3DHManager.
        prv_key = private_keys[0]
        pub_key = private_keys[1].public_key()
        x3dh = X3DHManager(prv_key=prv_key, trusted=[pub_key])
        # Verify that attributes match expectations.
        assert x3dh.id_key == prv_key
        assert isinstance(x3dh.sp_key, X25519PrivateKey)
        assert x3dh.trusted == {pub_key.public_bytes_raw()}
        assert isinstance(x3dh.secrets, dict) and not x3dh.secrets

    def test_create_handshake_request(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test that `create_handshake_request` outputs expected material."""
        # Setup a X3DHManager and create a handshake request.
        prv_key = private_keys[0]
        pub_key = private_keys[1].public_key()
        x3dh = X3DHManager(prv_key=prv_key, trusted=[pub_key])
        request = x3dh.create_handshake_request()
        # Verify that the request can be converted to bytes and contains
        # expected public keys. Leave further checks to further tests.
        assert isinstance(request, int)
        r_bytes = request.to_bytes(160, "big")
        assert x3dh.id_key.public_key().public_bytes_raw() in r_bytes
        assert x3dh.sp_key.public_key().public_bytes_raw() in r_bytes

    # Tests for request processing by a peer.

    def test_process_handshake_request(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test that `create_handshake_request` outputs expected material."""
        # Setup a pair of mutually-trusting X3DHManager instances.
        peer_a = X3DHManager(
            prv_key=private_keys[0], trusted=[private_keys[1].public_key()]
        )
        peer_b = X3DHManager(
            prv_key=private_keys[1], trusted=[private_keys[0].public_key()]
        )
        # Create a request and have it processed by the peer.
        request = peer_a.create_handshake_request()
        response = peer_b.process_handshake_request(request)
        # Verify that this results in a secret key being held by peer B.
        id_a = peer_a.id_key.public_key().public_bytes_raw()
        secret = peer_b.secrets.get(id_a, None)
        assert isinstance(secret, bytes) and len(secret) == 32
        # Verify that the response can be converted to bytes and contains
        # expected public keys. Leave further checks to further tests.
        assert isinstance(response, int)
        r_bytes = response.to_bytes(124, "big")
        assert peer_b.id_key.public_key().public_bytes_raw() in r_bytes
        assert request.to_bytes(160, "big")[-32:] in r_bytes

    def test_process_handshake_request_fails_invalid_inputs(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test that `process_handshake_request` raises on invalid inputs."""
        # Setup a X3DHManager.
        x3dh = X3DHManager(prv_key=private_keys[0], trusted=[])
        # Test that processing fails on improper request type.
        with pytest.raises(TypeError):
            x3dh.process_handshake_request("unproper-type")  # type: ignore
        # Test that processing fails on a request of improper size.
        request_too_big = secrets.randbits(160 * 10)
        with pytest.raises(TypeError):
            x3dh.process_handshake_request(request_too_big)
        # Test that processing fails on a proper-size random-valued request
        # (due to signature being inherently wrong).
        request_random = secrets.randbits(160 * 8)
        with pytest.raises(ValueError):
            x3dh.process_handshake_request(request_random)

    def test_process_handshake_request_fails_untrusted_peer(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test that `process_handshake_request` raises for untrusted peer."""
        # Setup a pair of mutually-unknown X3DHManager instances.
        peer_a = X3DHManager(prv_key=private_keys[0], trusted=[])
        peer_b = X3DHManager(prv_key=private_keys[1], trusted=[])
        # Verify that processing the request results in a KeyError.
        request = peer_a.create_handshake_request()
        with pytest.raises(KeyError):
            peer_b.process_handshake_request(request)

    def test_process_handshake_request_fails_invalid_signature(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test that `process_handshake_request` verifies key signature."""
        # Setup a pair of mutually-trusting X3DHManager instances.
        peer_a = X3DHManager(
            prv_key=private_keys[0], trusted=[private_keys[1].public_key()]
        )
        peer_b = X3DHManager(
            prv_key=private_keys[1], trusted=[private_keys[0].public_key()]
        )
        # Setup a request that comprises an unproper signature (as though
        # an attacker was trying to usurp peer A's identity).
        r_bytes = peer_a.create_handshake_request().to_bytes(160, "big")
        r_bytes = r_bytes[:64] + r_bytes[:64] + r_bytes[128:]
        request = int.from_bytes(r_bytes, "big")
        # Verify that processing the request results in a ValueError.
        with pytest.raises(ValueError):
            peer_b.process_handshake_request(request)

    # Tests for response-to-request processing by the first peer.

    def test_complete_x3dh_setup(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test the full round of X3DH setup between trusting peers."""
        # Setup a pair of mutually-trusting X3DHManager instances.
        peer_a = X3DHManager(
            prv_key=private_keys[0], trusted=[private_keys[1].public_key()]
        )
        peer_b = X3DHManager(
            prv_key=private_keys[1], trusted=[private_keys[0].public_key()]
        )
        # Run the full round of X3DH setup request and response.
        request = peer_a.create_handshake_request()
        response = peer_b.process_handshake_request(request)
        peer_a.process_handshake_response(response)
        # Verify that this results in a shared 32-bytes secret key.
        id_a = peer_a.id_key.public_key().public_bytes_raw()
        id_b = peer_b.id_key.public_key().public_bytes_raw()
        assert peer_a.secrets[id_b] == peer_b.secrets[id_a]
        skey = peer_a.secrets[id_b]
        assert isinstance(skey, bytes) and len(skey) == 32

    def test_process_handshake_response_fails_invalid_inputs(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test that `process_handshake_response` raises on invalid inputs."""
        # Setup a X3DHManager.
        x3dh = X3DHManager(prv_key=private_keys[0], trusted=[])
        # Test that processing fails on improper request type.
        with pytest.raises(TypeError):
            x3dh.process_handshake_response("unproper-type")  # type: ignore
        # Test that processing fails on a request of improper size.
        response_too_big = secrets.randbits(124 * 10)
        with pytest.raises(TypeError):
            x3dh.process_handshake_response(response_too_big)
        # Test that processing fails on a proper-size random-valued request
        # (due to the one-time key not matching any known value).
        response_random = secrets.randbits(124 * 8)
        with pytest.raises(KeyError):
            x3dh.process_handshake_response(response_random)

    def test_process_handshake_response_fails_untrusted_peer(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test that `process_handshake_response` raises for untrusted peer."""
        # Setup a pair of X3DHManager instances, where B trusts A
        # but not the other way around.
        peer_a = X3DHManager(prv_key=private_keys[0], trusted=[])
        peer_b = X3DHManager(
            prv_key=private_keys[1], trusted=[private_keys[0].public_key()]
        )
        # Have B process a request from A and emit a response.
        request = peer_a.create_handshake_request()
        response = peer_b.process_handshake_request(request)
        # Verify that processing the response results in a KeyError.
        with pytest.raises(KeyError):
            peer_a.process_handshake_response(response)

    def test_process_handshake_response_fails_invalid_identity(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test that `process_handshake_response` verifies secret validity."""
        # Setup a pair of X3DHManager instances, where B trusts A,
        # A does not trust B but trusts another peer's public key.
        peer_a = X3DHManager(
            prv_key=private_keys[0],
            trusted=[private_keys[1].public_key()],
        )
        peer_b = X3DHManager(
            prv_key=Ed25519PrivateKey.generate(),
            trusted=[private_keys[0].public_key()],
        )
        # Have peer B process a request from peer A, but lie on their identity.
        request = peer_a.create_handshake_request()
        response = peer_b.process_handshake_request(request)
        r_bytes = response.to_bytes(124, "big")
        id_key = private_keys[1].public_key().public_bytes_raw()
        r_bytes = id_key + r_bytes[32:]
        response = int.from_bytes(r_bytes, "big")
        # Verify that processing the responses results in a ValueError.
        with pytest.raises(ValueError):
            peer_a.process_handshake_response(response)

    def test_process_handshake_response_fails_onetimekey_reuse(
        self,
        private_keys: Tuple[Ed25519PrivateKey, Ed25519PrivateKey],
    ) -> None:
        """Test that `process_handshake_response` does not accept otk reuse."""
        # Setup a triplet of mutually-trusting X3DHManager instances.
        idk_a, idk_b = private_keys
        idk_c = Ed25519PrivateKey.generate()
        peer_a = X3DHManager(
            prv_key=idk_a, trusted=[idk_b.public_key(), idk_c.public_key()]
        )
        peer_b = X3DHManager(
            prv_key=idk_b, trusted=[idk_a.public_key(), idk_c.public_key()]
        )
        peer_c = X3DHManager(
            prv_key=idk_c, trusted=[idk_a.public_key(), idk_b.public_key()]
        )
        # Have peer A emit a single request, and both B and C process it.
        request = peer_a.create_handshake_request()
        reply_b = peer_b.process_handshake_request(request)
        reply_c = peer_c.process_handshake_request(request)
        # Verify that peer A properly processes the first reply they get.
        # A and B now hold a shared secret, distinct from that held by C.
        peer_a.process_handshake_response(reply_b)
        assert (
            peer_a.secrets[idk_b.public_key().public_bytes_raw()]
            == peer_b.secrets[idk_a.public_key().public_bytes_raw()]
            != peer_c.secrets[idk_a.public_key().public_bytes_raw()]
        )
        # Verify that peer C's response is rejected due to the one-time
        # pre-key having been erased upon finalization of its first use.
        with pytest.raises(KeyError):
            peer_a.process_handshake_response(reply_c)
