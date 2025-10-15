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

"""Extended Triple Diffie-Hellman (X3DH) key agreement protocol."""

import hashlib
import os
from typing import Dict, List, Optional, Tuple

import gmpy2  # type: ignore[import-untyped]
from cryptography import exceptions as cryptography_exceptions
from cryptography.hazmat.primitives import hashes as cryptography_hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers import aead as cryptography_aead
from cryptography.hazmat.primitives.kdf import hkdf as cryptography_hkdf

__all__ = [
    "X3DHManager",
]


class X3DHManager:
    """X3DH (Extended Triple Diffie-Hellman) key agreement manager.

    This class enables emitting and processing handshake messages
    to perform X3DH across peers, that results in pairwise secret
    symmetric encryption keys compatible with various algorithms.

    It uses Elliptic Curve 25519 keys (both for static identity,
    session pre-key and one-time pre-keys).

    A public-key infrastructure with X25519 keys is used in order
    to prevent man-in-the-middle attacks, so that X3DH setup may
    only succeed when both peers have previously exchanged public
    identity keys over some external trusted canal.

    This implementation is based on the Signal documentation of
    the [X3DH](https://www.signal.org/docs/specifications/x3dh/)
    protocol, which may be consulted for reference specification
    and auditing of the present code.
    """

    def __init__(
        self,
        prv_key: Ed25519PrivateKey,
        trusted: List[Ed25519PublicKey],
    ) -> None:
        """Instantiate the X3DH handshake manager.

        Parameters
        ----------
        prv_key:
            Private Ed25519 key acting as a static identity key.
            Its public key must be known to and trusted by peers
            for them to accept X3DH requests from this instance.
        trusted:
            List of public Ed25519 keys acting as trusted static
            identity keys from peers. Incoming X3DH requests or
            responses may only succeed if they come from peers
            whose identity key matches one of the trusted keys.
        """
        self.id_key = prv_key
        self.sp_key = X25519PrivateKey.generate()
        self.trusted = {key.public_bytes_raw() for key in trusted}
        self._pre_rq: Optional[bytes] = None
        self._otkeys: Dict[bytes, X25519PrivateKey] = {}
        self.secrets: Dict[bytes, bytes] = {}

    def _create_prerequest(
        self,
    ) -> bytes:
        """Return a bytes sequence constituting a base for requests."""
        # Gather the public keys and signature that are to be sent.
        id_key_pub = self.id_key.public_key().public_bytes_raw()
        sp_key_pub = self.sp_key.public_key().public_bytes_raw()
        sp_key_sig = self.id_key.sign(sp_key_pub)
        # Return a pre-request, that only lacks a one-time pre-key.
        return id_key_pub + sp_key_pub + sp_key_sig

    @staticmethod
    def _convert_private_ed25519_to_x25519(
        key: Ed25519PrivateKey,
    ) -> X25519PrivateKey:
        """Convert an Ed25519 private key to a X25519 one."""
        ed_key = hashlib.sha512(key.private_bytes_raw()).digest()[:32]
        rbytes = bytearray(ed_key)
        rbytes[0] &= 248
        rbytes[31] &= 127
        return X25519PrivateKey.from_private_bytes(rbytes)

    @staticmethod
    def _convert_public_ed25519_to_x25519(
        key: Ed25519PublicKey,
    ) -> X25519PublicKey:
        """Convert en Ed25519 public key to a X25519 one."""
        p_mod = gmpy2.mpz(2) ** 255 - 19
        y_val = int.from_bytes(key.public_bytes_raw(), "little", signed=True)
        y_msk = gmpy2.mod(y_val, 2**255)
        u_val = gmpy2.mod((1 + y_msk) * gmpy2.invert(1 - y_msk, p_mod), p_mod)
        ubytes = int(u_val).to_bytes(32, "little", signed=True)
        return X25519PublicKey.from_public_bytes(ubytes)

    def create_handshake_request(
        self,
    ) -> int:
        """Return a handshake bundle of public keys for X3DH setup.

        - Generate and record an ephemeral pair of one-time pre-keys.
        - Return a bundle of public keys that a peer may use for X3DH.

        Returns
        -------
        request:
            Int-converted 160-bytes sequence comprising (in that order)
            this instance's public identity key, signed pre-key, the
            signature of the latter, and a one-time public pre-key.
        """
        # Optionally create the static part of the request key bundle.
        if self._pre_rq is None:
            self._pre_rq = self._create_prerequest()
        # Create and record a one-time pre-key.
        ot_key = X25519PrivateKey.generate()
        ot_key_pub = ot_key.public_key().public_bytes_raw()
        self._otkeys[ot_key_pub] = ot_key
        # Add the public one-time pre-key to the bundle and convert it to int.
        return int.from_bytes(self._pre_rq + ot_key_pub, "big")

    def process_handshake_request(
        self,
        request: int,
    ) -> int:
        """Process a received handshake key bundle for X3DH setup.

        - Parse and verify received public keys.
        - Generate an ephemeral pair of keys.
        - Run the X3DH steps and derive a shared secret key.
        - Record that secret key under the peer's identity.
        - Return a bundle of public keys to send back to the peer.

        Parameters
        ----------
        request:
            Integer-converted bytes sequence of public keys emitted
            by a peer's `create_handshake_request` method.

        Returns
        -------
        response:
            Int-converted 124-bytes sequence comprising (in that order)
            this instance's public identity key, an ephemeral public key,
            the public one-time pre-key received from the peer, an AEAD
            encrypted message and the associated nonce.

        Raises
        ------
        KeyError:
            If the received identity key is not part of trusted keys.
        TypeError:
            If the input `request` cannot be parsed as expected.
        ValueError:
            If the received signed pre-key's signature does not match
            the received identity key.
        """
        # Gather peer public keys, and generate an ephemeral key.
        id_key, sp_key, ot_key = self._parse_and_verify_request(request)
        self._raise_if_identity_is_not_trusted(id_key)
        self_ek_key = X25519PrivateKey.generate()
        # Convert the identity keys for key-exchange purposes.
        self_ix_key = self._convert_private_ed25519_to_x25519(self.id_key)
        ix_key = self._convert_public_ed25519_to_x25519(id_key)
        # Run the Diffie-Hellman steps.
        dh_key = (
            self_ix_key.exchange(sp_key)
            + self_ek_key.exchange(ix_key)
            + self_ek_key.exchange(sp_key)
            + self_ek_key.exchange(ot_key)
        )
        # Run the derivation algorithm and record the resulting shared key.
        self._derive_and_record_shared_secret(id_key, dh_key)
        # Set up information to be sent back.
        id_key_pub = self.id_key.public_key().public_bytes_raw()
        ek_key_pub = self_ek_key.public_key().public_bytes_raw()
        ot_key_pub = ot_key.public_bytes_raw()  # signal which ot_key was used
        enc_aead_m = self._setup_aead_message(id_key)
        # Format it all as a single integer value.
        return int.from_bytes(
            id_key_pub + ek_key_pub + ot_key_pub + enc_aead_m, "big"
        )

    def _parse_and_verify_request(
        self,
        request: int,
    ) -> Tuple[Ed25519PublicKey, X25519PublicKey, X25519PublicKey]:
        """Parse received bundled keys, verifying integrity."""
        try:
            # Convert the request back to a bytes sequence.
            keys = request.to_bytes(160, "big")
            # Recover public identity key, signed pre-key and one-time pre-key.
            id_key = Ed25519PublicKey.from_public_bytes(keys[:32])
            sp_key = X25519PublicKey.from_public_bytes(keys[32:64])
            ot_key = X25519PublicKey.from_public_bytes(keys[128:])
        except Exception as exc:
            raise TypeError(
                "Failed to parse the input integer-formatted keys bundle."
            ) from exc
        # Verify that the second key's signature matches the identity key.
        try:
            id_key.verify(signature=keys[64:128], data=keys[32:64])
        except cryptography_exceptions.InvalidSignature as exc:
            raise ValueError(
                "The received signed pre-key's signature does not match"
                " the received identity key."
            ) from exc
        return id_key, sp_key, ot_key

    def _raise_if_identity_is_not_trusted(
        self,
        id_key: Ed25519PublicKey,
    ) -> None:
        """Raise a KeyError if an input ed25519 public key is not trusted."""
        if id_key.public_bytes_raw() not in self.trusted:
            raise KeyError(
                "None of the trusted public identity keys matches received "
                "identity key."
            )

    def _derive_and_record_shared_secret(
        self,
        id_key: Ed25519PublicKey,
        dh_key: bytes,
    ) -> None:
        """Run the derivation algorithm and record the resulting shared key."""
        hdkf = cryptography_hkdf.HKDF(
            cryptography_hashes.SHA512(), length=32, salt=None, info=None
        )
        shared = hdkf.derive(dh_key)
        self.secrets[id_key.public_bytes_raw()] = shared

    def _setup_aead_message(
        self,
        id_key: Ed25519PublicKey,
    ) -> bytes:
        """Set up a conventional AEAD message for a given peer."""
        self_id = self.id_key.public_key().public_bytes_raw()
        peer_id = id_key.public_bytes_raw()
        chacha = cryptography_aead.ChaCha20Poly1305(self.secrets[peer_id])
        nonce = os.urandom(12)
        enc_m = chacha.encrypt(nonce, b"", self_id + peer_id)
        return enc_m + nonce

    def _verify_aead_message(
        self,
        id_key: Ed25519PublicKey,
        aead_msg: bytes,
    ) -> None:
        """Verify a conventional AEAD message from a given peer."""
        self_id = self.id_key.public_key().public_bytes_raw()
        peer_id = id_key.public_bytes_raw()
        chacha = cryptography_aead.ChaCha20Poly1305(self.secrets[peer_id])
        enc_m = aead_msg[:16]
        nonce = aead_msg[16:]
        try:
            chacha.decrypt(nonce, enc_m, peer_id + self_id)
        except cryptography_exceptions.InvalidTag as exc:
            raise ValueError(
                "Verification of the AEAD part of the handshake response"
                " failed. This means the shared secret key setup failed,"
                " either due to some error somewhere or to some tampering."
            ) from exc

    def process_handshake_response(
        self,
        response: int,
    ) -> None:
        """Process a received key bundle in response to an initial sent one.

        - Parse received public keys and retrieve a private one-time key.
        - Run the X3DH steps and derive a shared secret key.
        - Record that secret key under the peer's identity.
        - Discard the ephemeral one-time key used for this X3DH setup.

        Parameters
        ----------
        response:
            Integer-converted bytes sequence of public keys output
            by a peer's `process_handshake_request` method called
            on a request generated by this instance.

        Raises
        ------
        KeyError
            If the one-time public key part of the bundle is not
            (or no longer) known to this instance.
        TypeError
            If the input `response` cannot be parsed as expected.
        ValueError
            If the AEAD (authenticated encryption associated data)
            part of the response cannot be verified.
        """
        # Parse received bundled keys and retrieve the one-time pre-key.
        id_key, ek_key, self_ot_key, aead_msg = self._parse_response_data(
            response
        )
        # Raise if the peer's identity key is not among trusted ones.
        self._raise_if_identity_is_not_trusted(id_key)
        # Convert Ed25519 identity keys to X25519 for key exchange.
        self_ix_key = self._convert_private_ed25519_to_x25519(self.id_key)
        ix_key = self._convert_public_ed25519_to_x25519(id_key)
        # Run the Diffie-Hellman steps.
        dh_1 = self.sp_key.exchange(ix_key)
        dh_2 = self_ix_key.exchange(ek_key)
        dh_3 = self.sp_key.exchange(ek_key)
        dh_4 = self_ot_key.exchange(ek_key)
        # Run the derivation algorithm and record the resulting shared key.
        dh_key = dh_1 + dh_2 + dh_3 + dh_4
        self._derive_and_record_shared_secret(id_key, dh_key)
        # Verify that the received AEAD-encrypted message matches,
        # indicating that the shared key matches between peers.
        self._verify_aead_message(id_key, aead_msg)

    def _parse_response_data(
        self,
        response: int,
    ) -> Tuple[Ed25519PublicKey, X25519PublicKey, X25519PrivateKey, bytes]:
        """Parse received bundled keys and message as part of a response."""
        try:
            # Convert the bundle back to a bytes sequence.
            keys = response.to_bytes(124, "big")
            # Recover the identity and ephemeral public keys.
            id_key = Ed25519PublicKey.from_public_bytes(keys[:32])
            ek_key = X25519PublicKey.from_public_bytes(keys[32:64])
        except Exception as exc:
            raise TypeError(
                "Failed to parse the input integer-formatted keys bundle."
            ) from exc
        # Retrieve the private one-time pre-key based on the public one.
        ot_key = self._otkeys.pop(keys[64:96], None)
        if ot_key is None:
            raise KeyError(
                "Received a X3DH response that uses a one-time pre-key which"
                " is not (or no longer) known to this instance. Either some"
                " information was misplaced (or tampered with), or the peer"
                " tried to re-use an exhausted one-time pre-key."
            )
        # Extract the AEAD-encrypted message and return it with public keys.
        aead_m = keys[96:124]
        return id_key, ek_key, ot_key, aead_m
