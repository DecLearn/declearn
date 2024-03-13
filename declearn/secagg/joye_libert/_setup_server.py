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

"""Server-side code for Joye-Libert SecAgg setup."""

import math
from typing import Dict, List, Optional, Set, Tuple


from declearn.communication.api import NetworkServer
from declearn.communication.utils import verify_client_messages_validity
from declearn.messaging import Error
from declearn.secagg.joye_libert._decrypt import JoyeLibertDecrypter
from declearn.secagg.joye_libert.messages import (
    JoyeLibertInitInfo,
    JoyeLibertPeerInfo,
    JoyeLibertPublicShare,
    JoyeLibertSecretShares,
    JoyeLibertShamirPrime,
)
from declearn.secagg.shamir import recover_shared_secret
from declearn.secagg.utils import generate_random_prime
from declearn.secagg.x3dh import run_x3dh_setup_server

__all__ = [
    "run_joye_libert_setup_server",
]


async def run_joye_libert_setup_server(
    netwk: NetworkServer,
    clients: Optional[Set[str]] = None,
    bitsize: int = 64,
    clipval: float = 1e5,
) -> JoyeLibertDecrypter:
    """Orchestrate a Joye-Libert SecAgg setup protocol.

    This routine starts with emitting messages that are to trigger
    the `declearn.secagg.joye_libert.run_joye_libert_setup_server`
    counterpart routine for all (selected) clients, that are made
    to result in the setup of properly-parametrized Joye-Libert
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


    Parameters
    ----------
    netwk:
        `NetworkServer` communication endpoint, that is expected
        to be running and already have clients registered to it
        when this instance's `async_run` method is called.
    clients:
        Optional subset of clients to which to restrict the setup.
    bitsize:
        Quantization hyper-parameter, defining the range of output
        quantized integers.
    clipval:
        Quantization hyper-parameter, defining a maximum absolute
        value for floating point numbers being (un)quantized.

    Returns
    -------
    decrypter:
        Joye-Libert decryption controller, parametrized to match
        clients' enrypter instances.
    """
    routine = ServerJoyeLibertSetup(netwk, bitsize, clipval)
    return await routine.async_run(clients)


class ServerJoyeLibertSetup:  # pylint: disable=too-few-public-methods
    """Server-side routine for the setup of Joye-Libert-based SecAgg.

    This class defines a routine that is to be run in parallel to that
    of `declearn.secagg.setup.ClientJoyeLibertSetup`, that results in
    the setup of properly-parametrized Joye-Libert secure aggregation
    controllers.
    """

    def __init__(
        self,
        netwk: NetworkServer,
        bitsize: int = 64,
        clipval: float = 1e5,
    ) -> None:
        """Instantiate the server-side setup routine.

        Parameters
        ----------
        netwk:
            `NetworkServer` communication endpoint, that is expected
            to be running and already have clients registered to it
            when this instance's `async_run` method is called.
        bitsize:
            Quantization hyper-parameter, defining the range of output
            quantized integers.
        clipval:
            Quantization hyper-parameter, defining a maximum absolute
            value for floating point numbers being (un)quantized.
        """
        self.netwk = netwk
        self.bitsize = bitsize
        self.clipval = clipval

    async def async_run(
        self,
        clients: Optional[Set[str]] = None,
    ) -> JoyeLibertDecrypter:
        """Run the Joye-Libert SecAgg setup routine.

        Parameters
        ----------
        clients:
            Optional subset of clients to which to restrict the setup.

        Returns
        -------
        decrypter:
            Joye-Libert decryption controller, parametrized to match
            clients' enrypter instances.
        """
        # Exchange pre-set hyperparameters and public id keys.
        biprime, id_keys = await self._exchange_hyperparameters(clients)
        # Have clients run X3DH to setup symmetric private key pairs.
        await run_x3dh_setup_server(netwk=self.netwk, clients=clients)
        # Orchestrate the generation and exchange of encrypted secret shares.
        prime = await self._exchange_shamir_secret_shares(id_keys, biprime)
        # Receive public secret shares and recover the Joye-Libert public key.
        public_key = await self._recover_public_key(id_keys, prime)
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
        clients: Optional[Set[str]] = None,
    ) -> Tuple[int, Dict[str, str]]:
        """Send quantization hyper-parameters. Receive biprime and id keys."""
        # Send initial request to clients.
        await self.netwk.broadcast_message(
            JoyeLibertInitInfo(bitsize=self.bitsize, clipval=self.clipval),
            clients=clients,
        )
        # Await public biprime and identity key from clients.
        # Ensure all clients share the same biprime key.
        # Record mappings between clients' identity key and name.
        received = await self.netwk.wait_for_messages(clients)
        messages = await verify_client_messages_validity(
            self.netwk, received, expected=JoyeLibertPeerInfo
        )
        biprime = 0
        id_keys = {}  # type: Dict[str, str]
        for client, msg in messages.items():
            if not biprime:
                biprime = msg.biprime
            elif biprime != msg.biprime:
                err_msg = "Clients disagree on the biprime number to use."
                await self.netwk.broadcast_message(Error(err_msg))
                raise RuntimeError(err_msg)
            id_keys[client] = msg.id_key
        # Return received information.
        return biprime, id_keys

    async def _exchange_shamir_secret_shares(
        self,
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
        prime = await self._generate_and_send_shamir_prime(clients, biprime)
        # Receive, dispatch and send back encrypted shares across clients.
        c_names = {val: key for key, val in id_keys.items()}
        received = await self.netwk.wait_for_messages(clients)
        messages = await verify_client_messages_validity(
            self.netwk, received, expected=JoyeLibertSecretShares
        )
        c_shares = {}  # type: Dict[str, Dict[str, str]]
        for client, msg in messages.items():
            for idk, val in msg.shares.items():
                c_shares.setdefault(c_names[idk], {})[id_keys[client]] = val
        messages = {
            client: JoyeLibertSecretShares(shares)
            for client, shares in c_shares.items()
        }
        await self.netwk.send_messages(messages)
        # Return the prime number used for Shamir algorithm.
        return prime

    async def _generate_and_send_shamir_prime(
        self,
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
        await self.netwk.broadcast_message(
            JoyeLibertShamirPrime(prime), clients=clients
        )
        return prime

    async def _recover_public_key(
        self,
        id_keys: Dict[str, str],
        prime: int,
    ) -> int:
        """Recover the public Joye-Libert key from public Shamir shares."""
        # Receive public Shamir secret shares from all peers.
        received = await self.netwk.wait_for_messages(clients=set(id_keys))
        messages = await verify_client_messages_validity(
            self.netwk, received, expected=JoyeLibertPublicShare
        )
        s_shares = []  # type: List[Tuple[int, int]]
        for client, msg in messages.items():
            x_coord = int.from_bytes(bytes.fromhex(id_keys[client]), "big")
            y_coord = msg.share
            s_shares.append((x_coord, y_coord))
        # Recover the public key for Joye-Libert decryption.
        return -recover_shared_secret(shares=s_shares, mprime=prime)
