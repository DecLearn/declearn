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

"""Controllers for Joye-Libert SecAgg setup."""

import base64
import dataclasses
import math
import secrets
from typing import Dict, List, Optional, Set, Tuple, Type, TypeVar

import cryptography.fernet
import gmpy2  # type: ignore

from declearn.communication.api import NetworkClient, NetworkServer
from declearn.communication.utils import (
    verify_client_messages_validity,
    verify_server_message_validity,
)
from declearn.messaging import Error, Message, SerializedMessage
from declearn.secagg.api import SecaggConfigClient, SecaggConfigServer
from declearn.secagg.joye_libert._decrypt import JoyeLibertDecrypter
from declearn.secagg.joye_libert._encrypt import JoyeLibertEncrypter
from declearn.secagg.joye_libert._primitives import DEFAULT_BIPRIME
from declearn.secagg.joye_libert.messages import (
    JoyeLibertPeerInfo,
    JoyeLibertPublicShare,
    JoyeLibertSecaggSetupQuery,
    JoyeLibertSecretShares,
    JoyeLibertShamirPrime,
)
from declearn.secagg.shamir import (
    generate_secret_shares,
    recover_shared_secret,
)
from declearn.secagg.utils import generate_random_prime
from declearn.secagg.x3dh import run_x3dh_setup_client, run_x3dh_setup_server

__all__ = [
    "JoyeLibertSecaggConfigClient",
    "JoyeLibertSecaggConfigServer",
]


MessageT = TypeVar("MessageT", bound=Message)


@dataclasses.dataclass
class JoyeLibertSecaggConfigClient(
    SecaggConfigClient[JoyeLibertEncrypter, JoyeLibertSecaggSetupQuery]
):
    """Client-side config and setup controller for Joye-Libert SecAgg.

    This class is two-fold:

    - On the one hand, it is a dataclass that can be parsed from TOML
      or kwargs, enabling to specify that Joye-Libert SecAgg should
      be used, and on what grounds.
    - On the other hand, its `setup_encrypter` async method should be
      called upon receiving a `JoyeLibertSecaggSetupQuery` from the
      server to participate in a Joye-Libert SecAgg setup protocol
      and return the resulting `JoyeLibertEncrypter`.

    Fields
    ------
    id_keys:
        `IdentityKeys` handler holding long-lived identity keys.
        This may be specified as a dict (notably in TOML files),
        with the following fields:
            - `prv_key`: path to a private ed25519 key.
            - `trusted`: path or list of paths to trusted peers'
              public ed25519 keys.
            - (opt.) `password`: optional password to decrypt the
              private key file; if required, a user prompt may be
              used rather than passing the password in clear.
    biprime:
        Public large biprime number defining the modulus for
        Joye-Libert operations. All peers must have defined
        the same value for the setup to succeed.

    Setup routine
    -------------
    The Joye-Libert SecAgg setup routine can be summarized as:

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
    across clients, so that setup messages can be verified to originate
    from trusted peers and not have been tampered with by the server.
    """

    biprime: int = DEFAULT_BIPRIME

    secagg_type = "joye-libert"

    async def _verify_server_message_validity(
        self,
        netwk: NetworkClient,
        received: SerializedMessage,
        expected: Type[MessageT],
    ) -> MessageT:
        try:
            return await verify_server_message_validity(
                netwk, received, expected
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("SecAgg setup protocol failed.") from exc

    async def setup_encrypter(
        self,
        netwk: NetworkClient,
        query: SerializedMessage[JoyeLibertSecaggSetupQuery],
    ) -> JoyeLibertEncrypter:
        # Exchange pre-set hyperparameters and public id keys.
        bitsize, clipval = await self._exchange_hyperparameters(netwk, query)
        # Run X3DH (Extended Triple Diffie-Hellman) to create ephemeral
        # pairwise symmetric encryption keys across clients.
        secret_peer_keys = await run_x3dh_setup_client(
            netwk, prv_key=self.id_keys.prv_key, trusted=self.id_keys.trusted
        )
        # Generate a private Joye-Libert key.
        secret_key = secrets.randbits(2 * self.biprime.bit_length())
        # Generate, encrypt and send secret shares of that key.
        share = await self._exchange_shamir_secret_shares(
            netwk, secret=secret_key, s_keys=secret_peer_keys
        )
        # Receive, decrypt and sum secret shares; send their public sum.
        await self._recover_public_share(
            netwk, share=share, s_keys=secret_peer_keys
        )
        # Instantiate and return a JoyeLibert crypter.
        return JoyeLibertEncrypter(
            prv_key=secret_key,
            biprime=self.biprime,
            bitsize=bitsize,
            clipval=clipval,
        )

    async def _exchange_hyperparameters(
        self,
        netwk: NetworkClient,
        received: SerializedMessage[JoyeLibertSecaggSetupQuery],
    ) -> Tuple[int, float]:
        """Receive quantization hyper-parameters. Send biprime and id key."""
        # Process initial message, containing quantization parameters.
        message = await self._verify_server_message_validity(
            netwk, received, expected=JoyeLibertSecaggSetupQuery
        )
        bitsize = message.bitsize
        clipval = message.clipval
        # Send back biprime number and public key.
        id_key = self.id_keys.pub_key_bytes.hex()
        await netwk.send_message(
            JoyeLibertPeerInfo(biprime=self.biprime, id_key=id_key)
        )
        # Return received information.
        return bitsize, clipval

    async def _exchange_shamir_secret_shares(
        self,
        netwk: NetworkClient,
        secret: int,
        s_keys: Dict[bytes, bytes],
    ) -> int:
        """Generate, encrypt and send shares of a secret key."""
        # Receive a public large prime number from the server.
        mprime = await self._receive_shamir_prime(netwk)
        # Split it into secret shares using Shamir algorithm.
        xcoord = {
            idk: int.from_bytes(idk, "big")
            for idk in {self.id_keys.pub_key_bytes, *s_keys}
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
        y_share = shares.pop(xcoord[self.id_keys.pub_key_bytes])
        # Encrypt, address and send secret shares to peers.
        id_keys = {coord: idk for idk, coord in xcoord.items()}
        peer_shares: Dict[str, str] = {}
        for coord, share in shares.items():
            key = base64.urlsafe_b64encode(s_keys[id_keys[coord]])
            enc = cryptography.fernet.Fernet(key).encrypt(
                share.to_bytes(mprime.bit_length(), "big")
            )
            peer_shares[id_keys[coord].hex()] = enc.hex()
        await netwk.send_message(JoyeLibertSecretShares(shares=peer_shares))
        # Return the secret key, and the secret share kept local.
        return y_share

    async def _receive_shamir_prime(
        self,
        netwk,
    ) -> int:
        """Await a shared prime number for Shamir secret sharing."""
        # Await the server-emitted shared prime number.
        received = await netwk.recv_message()
        message = await self._verify_server_message_validity(
            netwk, received, expected=JoyeLibertShamirPrime
        )
        prime = message.prime
        # Verify that the received number is a large-enough prime.
        if not (
            gmpy2.is_prime(prime)
            and (prime.bit_length() > 2 * self.biprime.bit_length())
        ):
            err_msg = "Invalid prime number for Shamir Secret Sharing."
            await netwk.send_message(Error(err_msg))
            raise RuntimeError(err_msg)
        return prime

    async def _recover_public_share(
        self,
        netwk: NetworkClient,
        share: int,
        s_keys: Dict[bytes, bytes],
    ) -> None:
        """Receive, decrypt and sum secret shares; send their public sum."""
        # Await encrypted secret shares from peers (routed by the server).
        received = await netwk.recv_message()
        message = await self._verify_server_message_validity(
            netwk, received, expected=JoyeLibertSecretShares
        )
        # Iteratively decrypt and sum the received partial shares.
        for idk, val in message.shares.items():
            key = s_keys[bytes.fromhex(idk)]
            key = base64.urlsafe_b64encode(key)
            shr = cryptography.fernet.Fernet(key).decrypt(bytes.fromhex(val))
            share += int.from_bytes(shr, "big")
        # Send back the obtained public share.
        await netwk.send_message(JoyeLibertPublicShare(share))


@dataclasses.dataclass
class JoyeLibertSecaggConfigServer(
    SecaggConfigServer[JoyeLibertDecrypter, JoyeLibertSecaggSetupQuery]
):
    """Server-side config and setup controller for Joye-Libert SecAgg.

    This class is two-fold:

    - On the one hand, it is a dataclass that can be parsed from TOML
      or kwargs, enabling to specify that Joye-Libert SecAgg should
      be used, and on what grounds.
    - On the other hand, its `setup_decrypter` async method should be
      called to trigger a protocol involving (a subset of) clients to
      set up SecAgg controllers. This involves sending a setup query
      message that should be passed by clients to their counterpart
      controller's `setup_encrypter` method. In the end, a resulting
      `JoyeLibertDecrypter` will be returned.

    Fields
    ------
    bitsize:
        Quantization hyper-parameter, defining the range of output
        quantized integers.
    clipval:
        Quantization hyper-parameter, defining a maximum absolute
        value for floating point numbers being (un)quantized.

    Setup routine
    -------------
    The Joye-Libert SecAgg setup routine can be summarized as:

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
    """

    secagg_type = "joye-libert"

    def prepare_secagg_setup_query(
        self,
    ) -> JoyeLibertSecaggSetupQuery:
        return JoyeLibertSecaggSetupQuery(
            bitsize=self.bitsize,
            clipval=self.clipval,
        )

    async def finalize_secagg_setup(
        self,
        netwk: NetworkServer,
        clients: Optional[Set[str]] = None,
    ) -> JoyeLibertDecrypter:
        # Exchange pre-set hyperparameters and public id keys.
        biprime, id_keys = await self._exchange_hyperparameters(netwk, clients)
        # Have clients run X3DH to setup symmetric private key pairs.
        await run_x3dh_setup_server(netwk=netwk, clients=clients)
        # Orchestrate the generation and exchange of encrypted secret shares.
        prime = await self._exchange_shamir_secret_shares(
            netwk, id_keys, biprime
        )
        # Receive public secret shares and recover the Joye-Libert public key.
        public_key = await self._recover_public_key(netwk, id_keys, prime)
        # Instantiate and return a Joye-Libert Decrypter.
        return JoyeLibertDecrypter(
            pub_key=public_key,
            n_peers=len(id_keys),
            biprime=biprime,
            bitsize=self.bitsize,
            clipval=self.clipval,
        )

    async def _exchange_hyperparameters(
        self,
        netwk: NetworkServer,
        clients: Optional[Set[str]] = None,
    ) -> Tuple[int, Dict[str, str]]:
        """Receive biprime and id keys."""
        # Await public biprime and identity key from clients.
        received = await netwk.wait_for_messages(clients)
        messages = await verify_client_messages_validity(
            netwk, received, expected=JoyeLibertPeerInfo
        )
        # Ensure all clients share the same biprime key.
        # Record mappings between clients' identity key and name.
        biprime = 0
        id_keys: Dict[str, str] = {}
        for client, msg in messages.items():
            if not biprime:
                biprime = msg.biprime
            elif biprime != msg.biprime:
                err_msg = "Clients disagree on the biprime number to use."
                await netwk.broadcast_message(Error(err_msg))
                raise RuntimeError(err_msg)
            id_keys[client] = msg.id_key
        # Return received information.
        return biprime, id_keys

    async def _exchange_shamir_secret_shares(
        self,
        netwk: NetworkServer,
        id_keys: Dict[str, str],
        biprime: int,
    ) -> int:
        """Orchestrate the generation and exchange of encrypted secret shares.

        - Generate and send a large prime number to all clients.
        - Await clients' encrypted Shamir secret shares and dispatch them back.

        Return the prime number used for Shamir secret sharing.
        """
        # Generate and share a large prime number.
        clients = set(id_keys)
        prime = await self._generate_and_send_shamir_prime(
            netwk, clients, biprime
        )
        # Receive, dispatch and send back encrypted shares across clients.
        c_names = {val: key for key, val in id_keys.items()}
        received = await netwk.wait_for_messages(clients)
        messages = await verify_client_messages_validity(
            netwk, received, expected=JoyeLibertSecretShares
        )
        c_shares: Dict[str, Dict[str, str]] = {}
        for client, msg in messages.items():
            for idk, val in msg.shares.items():
                c_shares.setdefault(c_names[idk], {})[id_keys[client]] = val
        messages = {
            client: JoyeLibertSecretShares(shares)
            for client, shares in c_shares.items()
        }
        await netwk.send_messages(messages)
        # Return the prime number used for Shamir algorithm.
        return prime

    async def _generate_and_send_shamir_prime(
        self,
        netwk: NetworkServer,
        clients: Set[str],
        biprime: int,
    ) -> int:
        """Generate and share a large prime number for Shamir Secret Sharing.

        Knowing that clients' secret keys have a bit length of twice that of
        the public biprime number, choose the prime size to guarantee that
        the sum of secret keys cannot be larger than it.
        """
        bitsize = 2 * biprime.bit_length() + math.ceil(math.log2(len(clients)))
        prime = generate_random_prime(bitsize=bitsize + 1)
        await netwk.broadcast_message(
            JoyeLibertShamirPrime(prime), clients=clients
        )
        return prime

    async def _recover_public_key(
        self,
        netwk: NetworkServer,
        id_keys: Dict[str, str],
        prime: int,
    ) -> int:
        """Recover the public Joye-Libert key from public Shamir shares."""
        # Receive public Shamir secret shares from all peers.
        received = await netwk.wait_for_messages(clients=set(id_keys))
        messages = await verify_client_messages_validity(
            netwk, received, expected=JoyeLibertPublicShare
        )
        s_shares: List[Tuple[int, int]] = []
        for client, msg in messages.items():
            x_coord = int.from_bytes(bytes.fromhex(id_keys[client]), "big")
            y_coord = msg.share
            s_shares.append((x_coord, y_coord))
        # Recover the public key for Joye-Libert decryption.
        return -recover_shared_secret(shares=s_shares, mprime=prime)
