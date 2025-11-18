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

"""X3DH (Extended Triple Diffie-Hellman) setup routines."""

from typing import Dict, List, Optional, Set, Type, TypeVar

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from declearn.communication.api import NetworkClient, NetworkServer
from declearn.communication.utils import (
    verify_client_messages_validity,
    verify_server_message_validity,
)
from declearn.messaging import Error, Message, SerializedMessage
from declearn.secagg.x3dh._x3dh import X3DHManager
from declearn.secagg.x3dh.messages import (
    X3DHOkay,
    X3DHRequests,
    X3DHResponses,
    X3DHTrigger,
)

__all__ = [
    "run_x3dh_setup_client",
    "run_x3dh_setup_server",
]


async def run_x3dh_setup_server(
    netwk: NetworkServer,
    clients: Optional[Set[str]] = None,
    seed: Optional[int] = None,
) -> None:
    """Orchestrate a X3DH (Extended Triple Diffie-Hellman) protocol run.

    Use the provided network communication endpoint to have clients run
    the X3DH protocol, resulting in their detaining pairwise ephemeral
    symmetric encryption keys.

    The server merely acts as an initiator, and transmists messages, the
    contents of which cannot be used to reveal information on private
    keys. Any tampering with messages would result in protocol failure,
    due to sensitive information being encrypted and/or signed by peers.

    Parameters
    ----------
    netwk:
        `NetworkServer` instance, to which clients have already
        registered.
    clients:
        Optional subset of clients to which to restrict the X3DH
        setup. If None, use all clients registered to `netwk`.
    seed:
        Optional seed for the RNG that decides for each pair of
        peers which will initiate the X3DH setup request.
        This has no incidence whatsoever on the final state.

    Raises
    ------
    KeyError
        If any received identity key is not part of `trusted` keys.
    RuntimeError
        If the protocol fails due to clients not following expected steps
        or raising errors themselves.
    """
    routine = X3DHServerRound(netwk, seed)
    await routine.async_run(clients)


async def run_x3dh_setup_client(
    netwk: NetworkClient,
    prv_key: Ed25519PrivateKey,
    trusted: List[Ed25519PublicKey],
) -> Dict[bytes, bytes]:
    """Participate in a X3DH (Extended Triple Diffie-Hellman) protocol.

    Use the provided network communication endpoint to communicate with
    peers through the mediation of a central server, and participate in
    the X3DH protocol, resulting in the detainment of pairwise ephemeral
    symmetric encryption keys with other peers.

    This requires having shared in advance some long-lived identity keys
    with peers, so that setup messages can be verified to originate from
    trusted peers and not have been tampered with by the server, that is
    merely acting as an initiator and a message-transmitter.

    Parameters
    ----------
    netwk:
        `NetworkClient` instance, that is already connected to and
        registered with its server-side counterpart.
    prv_key:
        Private Ed25519 key acting as a static identity key.
        Its public key must be known to and trusted by peers.
    trusted:
        List of public Ed25519 keys acting as trusted static
        identity keys from peers. The protocol will fail if
        any peer's identiy key is not among trusted ones.

    Returns
    -------
    secret_keys:
        Peer-wise ephemeral symmetric encryption keys, formatted as a
        `{public_identity_key.public_bytes_raw(): secret_key}` dict.
        Secret keys are 32-bytes-long and may notably be encoded into
        url-safe base64-encoded bytes for use with Fernet encryption.

    Raises
    ------
    KeyError
        If a peer's public identity key is not `trusted`.
        If a one-time public key is wrongfully referenced.
    RuntimeError
        If the protocol fails due to the server or peers not following
        expected steps or raising errors themselves.
    ValueError
        If a signature verification fails, that may indicate tempering.
    """
    x3dhm = X3DHManager(prv_key, trusted)
    routine = X3DHClientRound(netwk, x3dhm)
    msg = await netwk.recv_message()
    await routine.async_run(msg)
    return x3dhm.secrets


MessageT = TypeVar("MessageT", bound=Message)


class X3DHServerRound:  # pylint: disable=too-few-public-methods
    """Server-side X3DH (Extended Triple Diffie-Hellman) setup routine."""

    def __init__(
        self,
        netwk: NetworkServer,
        seed: Optional[int] = None,
    ) -> None:
        """Instantiate the server-side X3DH routine runner.

        Parameters
        ----------
        netwk:
            NetworkServer instance, to which clients have already
            registered.
        seed:
            Optional seed for the RNG that decides for each pair
            of peers which will initiate the X3DH setup request.
            This has no incidence whatsoever on the final state.
        """
        self.netwk = netwk
        self.rng = np.random.default_rng(seed)

    async def async_run(
        self,
        clients: Optional[Set[str]] = None,
    ) -> None:
        """Run the X3DH setup across the network of peers.

        Parameters
        ----------
        clients:
            Optional subset of clients to restrict the X3DH setup
            to which. If None, use all clients registered to this
            instance's `NetworkServer` instance (`self.netwk`).
        """
        # Decide which client is the initiator for each pair of peers.
        requests = self._draw_requests_directions(clients)
        # Send instructions to clients so that they set up X3DH requests.
        await self._send_initial_requests(requests)
        # Gather X3DH requests and distribute them to their recipients.
        received = await self.netwk.wait_for_messages(clients)
        await self._transmit_x3dh_requests(requests, received)
        # Gather X3DH responses and distribute them to their recipients.
        received = await self.netwk.wait_for_messages(clients)
        await self._transmit_x3dh_responses(requests, received)
        # Verify that each and every client is done and okay.
        received = await self.netwk.wait_for_messages(clients)
        await self._verify_messages_validity(received, X3DHOkay)
        # If things went right, confirm it to all clients.
        await self.netwk.broadcast_message(X3DHOkay(), clients)

    async def _send_initial_requests(
        self,
        requests: Dict[str, List[str]],
    ) -> None:
        """Send an initial X3DH setup instruction to clients."""
        messages = {
            name: X3DHTrigger(n_reqs=len(peers))
            for name, peers in requests.items()
        }
        await self.netwk.send_messages(messages)

    def _draw_requests_directions(
        self,
        clients_subset: Optional[Set[str]] = None,
    ) -> Dict[str, List[str]]:
        """Define which clients are to send requests to which."""
        # Draw the appropriate number of Bernoulli samples.
        clients = list(clients_subset or self.netwk.client_names)
        n_cli = len(clients)
        n_req = (n_cli * (n_cli - 1)) // 2
        direction = self.rng.uniform(size=n_req) < 0.5
        # Format results as a dict: for each client, those they will request.
        requests: Dict[str, List[str]] = {name: [] for name in clients}
        idx = 0
        for cdx, cli_a in enumerate(clients[:-1], start=1):
            for cli_b in clients[cdx:]:
                if direction[idx]:
                    requests[cli_a].append(cli_b)
                else:
                    requests[cli_b].append(cli_a)
                idx += 1
        return requests

    async def _transmit_x3dh_requests(
        self,
        requests: Dict[str, List[str]],
        received: Dict[str, SerializedMessage],
    ) -> None:
        """Transmit X3DH requests from clients to their peers."""
        queries = await self._verify_messages_validity(received, X3DHRequests)
        # Receive requests from clients and dispatch them by recipient.
        replies = {client: X3DHRequests(requests=[]) for client in queries}
        for client, cli_msg in queries.items():
            assert len(cli_msg.requests) == len(requests[client])
            for dst, req in zip(
                requests[client], cli_msg.requests, strict=False
            ):
                replies[dst].requests.append(req)
        # Send back triaged requests to their recipients.
        await self.netwk.send_messages(replies)

    async def _transmit_x3dh_responses(
        self,
        requests: Dict[str, List[str]],
        received: Dict[str, SerializedMessage],
    ) -> None:
        """Transmit X3DH responses from clients to their peers."""
        queries = await self._verify_messages_validity(received, X3DHResponses)
        # Receive responses from clients and dispatch them by recipient.
        replies = {client: X3DHResponses(responses=[]) for client in queries}
        for dst, sources in requests.items():
            for src in sources:
                response = queries[src].responses.pop(0)
                replies[dst].responses.append(response)
        # Send back triaged responses to their recipients.
        await self.netwk.send_messages(replies)

    async def _verify_messages_validity(
        self,
        received: Dict[str, SerializedMessage],
        expected: Type[MessageT],
    ) -> Dict[str, MessageT]:
        """Send an Error message and raise if messages are unproper."""
        try:
            return await verify_client_messages_validity(
                self.netwk, received, expected
            )
        except Exception as exc:
            raise RuntimeError(
                "X3DH failed due to invalid messages being exchanged."
            ) from exc


class X3DHClientRound:  # pylint: disable=too-few-public-methods
    """Client-side X3DH (Extended Triple Diffie-Hellman) setup routine."""

    def __init__(
        self,
        netwk: NetworkClient,
        x3dhm: X3DHManager,
    ) -> None:
        """Instantiate the client-side X3DH routine runner.

        Parameters
        ----------
        netwk:
            NetworkClient instance, that is already connected to
            and registered with its server-side counterpart.
        x3dhm:
            X3DHManager instance, that holds this client's private
            identity key and a list of trusted peers' public keys,
            and will hold the established pairwise symmetric keys
            after this setup round has run.
        """
        self.netwk = netwk
        self.x3dhm = x3dhm

    async def async_run(
        self,
        msg: SerializedMessage,
    ) -> None:
        """Run the X3DH setup across the network of peers.

        Parameters
        ----------
        msg:
            X3DH setup initiating request received from the server.
        """
        # Process initial server instructions and send back X3DH requests.
        await self._create_x3dh_requests(msg)
        # Process X3DH requests from peers and send back responses.
        received = await self.netwk.recv_message()
        await self._respond_x3dh_requests(received)
        # Process X3DH responses from peers and send back final status.
        received = await self.netwk.recv_message()
        await self._process_x3dh_responses(received)
        # Await confirmation that things went right for all peers.
        received = await self.netwk.recv_message()
        await self._verify_message_validity(received, X3DHOkay)

    async def _create_x3dh_requests(
        self,
        received: SerializedMessage,
    ) -> None:
        """Setup and send X3DH requests to the server."""
        query = await self._verify_message_validity(received, X3DHTrigger)
        requests = [
            self.x3dhm.create_handshake_request() for _ in range(query.n_reqs)
        ]
        await self.netwk.send_message(X3DHRequests(requests))

    async def _respond_x3dh_requests(
        self,
        received: SerializedMessage,
    ) -> None:
        """Process X3DH requests received from the server."""
        query = await self._verify_message_validity(received, X3DHRequests)
        try:
            responses = [
                self.x3dhm.process_handshake_request(request)
                for request in query.requests
            ]
        except (KeyError, TypeError, ValueError) as exc:
            await self.netwk.send_message(Error(repr(exc)))
            raise exc
        await self.netwk.send_message(X3DHResponses(responses))

    async def _process_x3dh_responses(
        self,
        received: SerializedMessage,
    ) -> None:
        """Process X3DH responses received from the server."""
        query = await self._verify_message_validity(received, X3DHResponses)
        try:
            for response in query.responses:
                self.x3dhm.process_handshake_response(response)
        except (KeyError, TypeError, ValueError) as exc:
            await self.netwk.send_message(Error(repr(exc)))
            raise exc
        await self.netwk.send_message(X3DHOkay())

    async def _verify_message_validity(
        self,
        received: SerializedMessage,
        expected: Type[MessageT],
    ) -> MessageT:
        """Send an Error message and/or raise if a message is unproper."""
        try:
            return await verify_server_message_validity(
                self.netwk, received, expected
            )
        except Exception as exc:
            raise RuntimeError(
                "X3DH failed due to invalid messages being exchanged."
            ) from exc
