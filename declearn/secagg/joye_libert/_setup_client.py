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

"""Client-side code for Joye-Libert SecAgg setup."""

import base64
import secrets
from typing import Dict, List, Tuple, TypeVar

import cryptography.fernet
import gmpy2
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from declearn.communication.api import NetworkClient
from declearn.communication.utils import verify_server_message_validity
from declearn.messaging import Error, Message, SerializedMessage
from declearn.secagg.joye_libert._primitives import DEFAULT_BIPRIME
from declearn.secagg.joye_libert._encrypt import JoyeLibertEncrypter
from declearn.secagg.joye_libert.messages import (
    JoyeLibertInitInfo,
    JoyeLibertPeerInfo,
    JoyeLibertPublicShare,
    JoyeLibertSecretShares,
    JoyeLibertShamirPrime,
)
from declearn.secagg.shamir import generate_secret_shares
from declearn.secagg.x3dh import run_x3dh_setup_client

__all__ = [
    "ClientJoyeLibertSetup",
]


MessageT = TypeVar("MessageT", bound=Message)


async def run_joye_libert_setup_client(
    netwk: NetworkClient,
    prv_key: Ed25519PrivateKey,
    trusted: List[Ed25519PublicKey],
    biprime: int = DEFAULT_BIPRIME,
) -> JoyeLibertEncrypter:
    """Participate in a Joye-Libert SecAgg setup protocol.

    This routine is to be run in parallel to peer clients' one and to
    `declearn.secagg.joye_libert.run_joye_libert_setup_server` and is
    made to result in the setup of properly-parametrized Joye-Libert
    secure aggregation controllers for all participants.

    This routine can be summarized as:

    - Server and Clients exchange some hyper-parameters for quantization
      and Joye-Libert secure aggregation.
    - Clients run the X3DH (Extended Triple Diffie-Hellman) protocol to
      set up pairwise ephemeral symmetric encryption keys, with messages
      passing by the Server, and using pre-shared public identity keys.
    - Server generates a public prime number for Shamir Secret Sharing.
      Clients generate a private key for Joye-Libert encryption, split
      it into shares using Shamir at coordinates derived from peers'
      identity keys.
    - Clients exchange their encrypted secret shares, with messages again
      passing by the Server, so that each client receives, decrypts and
      sums secret shares at the coordinate matching its identity key.
    - Server receives the resulting public secret shares from the Clients,
      and threfore recovers the sum of clients' private keys using Shamir.
      The opposite of this sum defines the public Joye-Libert key.

    This requires having shared in advance some long-lived identity keys
    with peers, so that setup messages can be verified to originate from
    trusted peers and not have been tampered with by the server.

    Parameters
    ----------
    netwk:
        `NetworkClient` communication endpoint, that is expected
        to be running and already be registered to a server when
        this instance's `async_run` method is called.
    prv_key:
        Private Ed25519 key acting as a static identity key.
        Its public key must be known to and trusted by peers.
    trusted:
        List of public Ed25519 keys acting as trusted static
        identity keys from peers. All peers must use trusted
        keys for the setup to succeed.
    biprime:
        Public large biprime number defining the modulus for
        Joye-Libert operations. All peers must have defined
        the same value for the setup to succeed.

    Returns
    -------
    encrypter:
        Joye-Libert SecAgg encryption controller, set up to match
        peer clients' hyper-parameters and server's decrypter.

    Raises
    ------
    KeyError
        If a peer's public identity key is not `trusted`.
        If a temporary key is misused as part of the process.
    RuntimeError
        If the protocol fails due to the server or peers not following
        expected steps or raising errors themselves.
    ValueError
        If a signature verification fails, that may indicate tempering.

    Notes on `biprime`
    ------------------
    - As the biprime property of a number is (by design) hard to
        prove, this implementation requires clients to agree on a
        shared trusted value, that will be imposed to the server
        and verified to be consensual as part of this setup.
    - Clients' private keys will be set to have a bitsize equal
        to twice that of `biprime`. Therefore, the larger `biprime`,
        the more secure the keys, but also the heavier the encrypted
        values, resulting in higher communication overhead costs.
    - The biprime number should be larger that any sum of cleartext
        values being securely aggregated. In practice this should not
        be an issue, as encrypted values will typically be 32 or 64
        bits integers, whereas biprime is 1023-bits-large by default.
    """
    routine = ClientJoyeLibertSetup(netwk, prv_key, trusted, biprime)
    msg = await netwk.recv_message()
    return await routine.async_run(msg)


class ClientJoyeLibertSetup:  # pylint: disable=too-few-public-methods
    """Client-side routine for the setup of Joye-Libert-based SecAgg.

    This class defines a routine that is to be run in parallel to peer
    clients' one and to `declearn.secagg.setup.ServerJoyeLibertSetup`,
    resulting in the setup of properly-parametrized Joye-Libert secure
    aggregation controllers.
    """

    def __init__(
        self,
        netwk: NetworkClient,
        prv_key: Ed25519PrivateKey,
        trusted: List[Ed25519PublicKey],
        biprime: int = DEFAULT_BIPRIME,
    ) -> None:
        """Instantiate the client-side setup routine.

        Parameters
        ----------
        netwk:
            `NetworkClient` communication endpoint, that is expected
            to be running and already be registered to a server when
            this instance's `async_run` method is called.
        prv_key:
            Private Ed25519 key acting as a static identity key.
            Its public key must be known to and trusted by peers.
        trusted:
            List of public Ed25519 keys acting as trusted static
            identity keys from peers. All peers must use trusted
            keys for the setup to succeed.
        biprime:
            Public large biprime number defining the modulus for
            Joye-Libert operations. All peers must have defined
            the same value for the setup to succeed.
        """
        self.netwk = netwk
        self.prv_key = prv_key
        self.trusted = trusted
        self.id_key = self.prv_key.public_key().public_bytes_raw()
        self.biprime = biprime

    async def async_run(
        self,
        message: SerializedMessage[JoyeLibertInitInfo],
    ) -> JoyeLibertEncrypter:
        """Run the Joye-Libert SecAgg setup routine.

        Parameters
        ----------
        message:
            `SerializedMessage` wrapping a Joye-Libert setup
            request from the server.

        Returns
        -------
        encrypter:
            Joye-Libert encryption controller, parametrized to
            match peer clients' hyper-parameters and server's
            decrypter.

        Raises
        ------
        RuntimeError
            If the protocol does not go as expected.

        Exceptions from `run_x3dh_setup_client` may also be raised.
        """
        # Exchange pre-set hyperparameters and public id keys.
        bitsize, clipval = await self._exchange_hyperparameters(message)
        # Run X3DH (Extended Triple Diffie-Hellman) to create ephemeral
        # pairwise symmetric encryption keys across clients.
        secret_peer_keys = await run_x3dh_setup_client(
            self.netwk, self.prv_key, self.trusted
        )
        # Generate a private Joye-Libert key.
        secret_key = secrets.randbits(2 * self.biprime.bit_length())
        # Generate, encrypt and send secret shares of that key.
        share = await self._exchange_shamir_secret_shares(
            secret=secret_key, s_keys=secret_peer_keys
        )
        # Receive, decrypt and sum secret shares; send their public sum.
        await self._recover_public_share(share=share, s_keys=secret_peer_keys)
        # Instantiate and return a JoyeLibert crypter.
        return JoyeLibertEncrypter(
            prv_key=secret_key,
            biprime=self.biprime,
            bitsize=bitsize,
            clipval=clipval,
        )

    async def _exchange_hyperparameters(
        self,
        received: SerializedMessage[JoyeLibertInitInfo],
    ) -> Tuple[int, float]:
        """Receive quantization hyper-parameters. Send biprime and id key."""
        # Process initial message, containing quantization parameters.
        message = await verify_server_message_validity(
            self.netwk, received, expected=JoyeLibertInitInfo
        )
        bitsize = message.bitsize
        clipval = message.clipval
        # Send back biprime number and public key.
        await self.netwk.send_message(
            JoyeLibertPeerInfo(biprime=self.biprime, id_key=self.id_key.hex())
        )
        # Return received information.
        return bitsize, clipval

    async def _exchange_shamir_secret_shares(
        self,
        secret: int,
        s_keys: Dict[bytes, bytes],
    ) -> int:
        """Generate, encrypt and send shares of a secret key."""
        # Receive a public large prime number from the server.
        mprime = await self._receive_shamir_prime()
        # Split it into secret shares using Shamir algorithm.
        xcoord = {
            idk: int.from_bytes(idk, "big") for idk in {self.id_key, *s_keys}
        }
        shares = dict(
            generate_secret_shares(
                secret=secret,
                shares=len(xcoord),
                xcoord=list(xcoord.values()),
                mprime=mprime,
            )
        )
        # Extract the share that would be adressed back to this peer.
        y_share = shares.pop(xcoord[self.id_key])
        # Encrypt, address and send secret shares to peers.
        id_keys = {coord: idk for idk, coord in xcoord.items()}
        peer_shares = {}  # type: Dict[str, str]
        for coord, share in shares.items():
            key = base64.urlsafe_b64encode(s_keys[id_keys[coord]])
            enc = cryptography.fernet.Fernet(key).encrypt(
                share.to_bytes(mprime.bit_length(), "big")
            )
            peer_shares[id_keys[coord].hex()] = enc.hex()
        await self.netwk.send_message(
            JoyeLibertSecretShares(shares=peer_shares)
        )
        # Return the secret key, and the secret share kept local.
        return y_share

    async def _receive_shamir_prime(
        self,
    ) -> int:
        """Await a shared prime number for Shamir secret sharing."""
        # Await the server-emitted shared prime number.
        received = await self.netwk.recv_message()
        message = await verify_server_message_validity(
            self.netwk, received, expected=JoyeLibertShamirPrime
        )
        prime = message.prime
        # Verify that the received number is a large-enough prime.
        if not (
            gmpy2.is_prime(prime)
            and (prime.bit_length() > 2 * self.biprime.bit_length())
        ):
            err_msg = "Invalid prime number for Shamir Secret Sharing."
            await self.netwk.send_message(Error(err_msg))
            raise RuntimeError(err_msg)
        return prime

    async def _recover_public_share(
        self,
        share: int,
        s_keys: Dict[bytes, bytes],
    ) -> None:
        """Receive, decrypt and sum secret shares; send their public sum."""
        # Await encrypted secret shares from peers (routed by the server).
        received = await self.netwk.recv_message()
        message = await verify_server_message_validity(
            self.netwk, received, expected=JoyeLibertSecretShares
        )
        # Iteratively decrypt and sum the received partial shares.
        for idk, val in message.shares.items():
            key = s_keys[bytes.fromhex(idk)]
            key = base64.urlsafe_b64encode(key)
            shr = cryptography.fernet.Fernet(key).decrypt(bytes.fromhex(val))
            share += int.from_bytes(shr, "big")
        # Send back the obtained public share.
        await self.netwk.send_message(JoyeLibertPublicShare(share))
