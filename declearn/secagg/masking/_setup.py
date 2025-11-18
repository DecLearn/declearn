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

"""Controllers for Masking-based SecAgg setup."""

import dataclasses
from typing import Dict, List, Optional, Set, Tuple

from declearn.communication.api import NetworkClient, NetworkServer
from declearn.communication.utils import (
    verify_client_messages_validity,
    verify_server_message_validity,
)
from declearn.messaging import SerializedMessage
from declearn.secagg.api import SecaggConfigClient, SecaggConfigServer
from declearn.secagg.masking._decrypt import MaskingDecrypter
from declearn.secagg.masking._encrypt import MaskingEncrypter
from declearn.secagg.masking.messages import (
    MaskingSecaggSetupQuery,
    MaskingSecaggSetupReply,
)
from declearn.secagg.x3dh import run_x3dh_setup_client, run_x3dh_setup_server

__all__ = [
    "MaskingSecaggConfigClient",
    "MaskingSecaggConfigServer",
]


@dataclasses.dataclass
class MaskingSecaggConfigClient(
    SecaggConfigClient[MaskingEncrypter, MaskingSecaggSetupQuery]
):
    """Client-side config and setup controller for masking-based SecAgg.

    This class is two-fold:

    - On the one hand, it is a dataclass that can be parsed from TOML
      or kwargs, enabling to specify that Masking-based SecAgg should
      be used, and on what grounds.
    - On the other hand, its `setup_encrypter` async method should be
      called upon receiving a `MaskingSecaggSetupQuery` from the server
      to participate in a Maksing-based SecAgg setup protocol and return
      the resulting `MaskingEncrypter`.

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

    Setup routine
    -------------
    The Masking-based SecAgg setup routine can be summarized as:

    - Server sends a query, including quantization hyper-parameters.
    - Clients run the X3DH (Extended Triple Diffie-Hellman) protocol to
      set up pairwise ephemeral symmetric secret keys, with messages
      passing by the Server, and using pre-shared public identity keys.
    - Clients derive RNG seeds from the generated secret keys, that will
      be used to generate pseudo-random sum-canceling masking values.
    - Server merely sets up a decrypter matching the number of clients,
      after receiving confirmation from clients that setup went well.

    This requires having shared in advance some long-lived identity keys
    across clients, so that setup messages can be verified to originate
    from trusted peers and not have been tampered with by the server.
    """

    secagg_type = "masking"

    async def setup_encrypter(
        self,
        netwk: NetworkClient,
        query: SerializedMessage[MaskingSecaggSetupQuery],
    ) -> MaskingEncrypter:
        # Type-check and deserialize the received message.
        message = await verify_server_message_validity(
            netwk, received=query, expected=MaskingSecaggSetupQuery
        )
        # Respond that hyper-parameters were accepted.
        await netwk.send_message(MaskingSecaggSetupReply())
        # Run X3DH (Extended Triple Diffie-Hellman) to create ephemeral
        # pairwise secrets, that will be used as PRNG seeds for masks.
        secret_peer_keys = await run_x3dh_setup_client(
            netwk, prv_key=self.id_keys.prv_key, trusted=self.id_keys.trusted
        )
        # Derive secret PRNG masking seeds from secret symmetric keys.
        pos_masks_seeds, neg_masks_seeds = (
            self._setup_rng_seeds_from_symmetric_keys(secret_peer_keys)
        )
        # Instantiate a MaskingEncrypter.
        encrypter = MaskingEncrypter(
            pos_masks_seeds=pos_masks_seeds,
            neg_masks_seeds=neg_masks_seeds,
            bitsize=message.bitsize,
            clipval=message.clipval,
        )
        # Signal to the server that setup went fine, then return the encrypter.
        await netwk.send_message(MaskingSecaggSetupReply())
        return encrypter

    def _setup_rng_seeds_from_symmetric_keys(
        self,
        secret_peer_keys: Dict[bytes, bytes],
    ) -> Tuple[List[int], List[int]]:
        """Derive secret PRNG masking seeds from secret symmetric keys.

        Deterministically turn secret keys into PRNG seeds, then decide
        whether to make the associated mask an addition or substraction
        one in a pseudo-random way that is symmetric across peer pairs.
        """
        this_key = self.id_keys.prv_key.public_key().public_bytes_raw()
        pos_masks_seeds: List[int] = []
        neg_masks_seeds: List[int] = []
        for peer_key, peer_secret in secret_peer_keys.items():
            seed = int.from_bytes(peer_secret, "big")
            if (seed + (this_key < peer_key)) % 2:
                pos_masks_seeds.append(seed)
            else:
                neg_masks_seeds.append(seed)
        return pos_masks_seeds, neg_masks_seeds


@dataclasses.dataclass
class MaskingSecaggConfigServer(
    SecaggConfigServer[MaskingDecrypter, MaskingSecaggSetupQuery]
):
    """Server-side config and setup controller for masking-based SecAgg.

    This class is two-fold:

    - On the one hand, it is a dataclass that can be parsed from TOML
      or kwargs, enabling to specify that Masking-based SecAgg should
      be used, and on what grounds.
    - On the other hand, its `setup_decrypter` async method should be
      called to trigger a protocol involving (a subset of) clients to
      set up SecAgg controllers. This involves sending a setup query
      message that should be passed by clients to their counterpart
      controller's `setup_encrypter` method. In the end, a resulting
      `MaskingDecrypter` will be returned.

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
    The Masking-based SecAgg setup routine can be summarized as:

    - Server sends a query, including quantization hyper-parameters.
    - Clients run the X3DH (Extended Triple Diffie-Hellman) protocol to
      set up pairwise ephemeral symmetric secret keys, with messages
      passing by the Server, and using pre-shared public identity keys.
    - Clients derive RNG seeds from the generated secret keys, that will
      be used to generate pseudo-random sum-canceling masking values.
    - Server merely sets up a decrypter matching the number of clients,
      after receiving confirmation from clients that setup went well.
    """

    secagg_type = "masking"

    def prepare_secagg_setup_query(
        self,
    ) -> MaskingSecaggSetupQuery:
        return MaskingSecaggSetupQuery(
            bitsize=self.bitsize,
            clipval=self.clipval,
        )

    async def finalize_secagg_setup(
        self,
        netwk: NetworkServer,
        clients: Optional[Set[str]] = None,
    ) -> MaskingDecrypter:
        # Except clients to answer positively to the initial query.
        replies = await netwk.wait_for_messages(clients)
        try:
            await verify_client_messages_validity(
                netwk, replies, expected=MaskingSecaggSetupReply
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Masking-based SecAgg setup initialization failed."
            ) from exc
        # Run X3DH (Extended Triple Diffie-Hellman) to create ephemeral
        # pairwise secrets across clients, that will be used for masking.
        await run_x3dh_setup_server(netwk, clients)
        # Except an okay flag from all clients.
        replies = await netwk.wait_for_messages(clients)
        try:
            await verify_client_messages_validity(
                netwk, replies, expected=MaskingSecaggSetupReply
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Masking-based SecAgg setup finalization failed."
            ) from exc
        # Return a deterministically-setup decrypter.
        return MaskingDecrypter(
            n_peers=len(replies), bitsize=self.bitsize, clipval=self.clipval
        )
