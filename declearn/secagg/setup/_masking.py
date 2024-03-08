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

"""Routines for Masking-based SecAgg setup."""

from typing import List, Optional, Set

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from declearn.communication.api import NetworkClient, NetworkServer
from declearn.secagg.masking import MaskingDecrypter, MaskingEncrypter
from declearn.secagg.setup.messages import (
    MaskingSecaggSetupInit,
    MaskingSecaggSetupOkay,
)
from declearn.secagg.x3dh import run_x3dh_setup_client, run_x3dh_setup_server

__all__ = [
    "run_masking_secagg_setup_client",
    "run_masking_secagg_setup_server",
]


async def run_masking_secagg_setup_client(
    message: MaskingSecaggSetupInit,
    netwk: NetworkClient,
    prv_key: Ed25519PrivateKey,
    trusted: List[Ed25519PublicKey],
) -> MaskingEncrypter:
    """Participate in a protocol to set up masking-based secure aggregation."""
    # Respond that hyper-parameters were accepted.
    await netwk.send_message(MaskingSecaggSetupOkay())
    # Run X3DH (Extended Triple Diffie-Hellman) to create ephemeral
    # pairwise secrets, that will be used as PRNG seeds for masks.
    secret_peer_keys = await run_x3dh_setup_client(netwk, prv_key, trusted)
    # Decide for each PRNG whether to add or substract its resulting mask,
    # in a randomized and symmetric way across pairs of peers.
    this_key = prv_key.public_key().public_bytes_raw()
    pos_masks_seeds = []  # type: List[int]
    neg_masks_seeds = []  # type: List[int]
    for peer_key, peer_secret in secret_peer_keys.items():
        seed = int.from_bytes(peer_secret, "big")
        if (seed + (this_key < peer_key)) % 2:
            pos_masks_seeds.append(seed)
        else:
            neg_masks_seeds.append(seed)
    # Instantiate a MaskingEncrypter.
    encrypter = MaskingEncrypter(
        pos_masks_seeds=pos_masks_seeds,
        neg_masks_seeds=neg_masks_seeds,
        bitsize=message.bitsize,
        clipval=message.clipval,
    )
    # Signal to the server that setup went fine, then return the encrypter.
    await netwk.send_message(MaskingSecaggSetupOkay())
    return encrypter


async def run_masking_secagg_setup_server(
    netwk: NetworkServer,
    clients: Optional[Set[str]] = None,
    bitsize: int = 64,
    clipval: float = 1e5,
) -> MaskingDecrypter:
    """Orchestrate a protocol to set up masking-based secure aggregation."""
    # Send initial request to clients and expect an okay flag.
    await netwk.broadcast_message(
        MaskingSecaggSetupInit(bitsize=bitsize, clipval=clipval),
        clients=clients,
    )
    replies = await netwk.wait_for_messages(clients)
    failed = "\n".join(
        f"\tClient '{client}': message with type '{reply.message_cls}'"
        for client, reply in replies.items()
        if not issubclass(reply.message_cls, MaskingSecaggSetupOkay)
    )
    if failed:
        raise RuntimeError(
            "MaskingSecagg setup failed: some clients replied with an "
            f"unexpected message to the initial setup message:\n{failed}"
        )
    # Run X3DH (Extended Triple Diffie-Hellman) to create ephemeral
    # pairwise secrets across clients, that will be used for masking.
    await run_x3dh_setup_server(netwk, clients)
    # Except an okay flag from all clients.
    replies = await netwk.wait_for_messages(clients)
    failed = "\n".join(
        f"\tClient '{client}': message with type '{reply.message_cls}'"
        for client, reply in replies.items()
        if not issubclass(reply.message_cls, MaskingSecaggSetupOkay)
    )
    if failed:
        raise RuntimeError(
            "MaskingSecagg setup failed: some clients replied with an "
            f"unexpected final message:\n{failed}"
        )
    # Return a decrypter.
    return MaskingDecrypter(
        n_peers=len(replies), bitsize=bitsize, clipval=clipval
    )
