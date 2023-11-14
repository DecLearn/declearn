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

"""X3DH (Extended Triple Diffie-Hellman) setup routines."""

import asyncio
from typing import Dict, List, Optional, Set

import numpy as np

from declearn.communication import messaging
from declearn.communication.api import NetworkClient, NetworkServer
from declearn.secagg.x3dh._x3dh import X3DHManager


__all__ = [
    "X3DHClientRound",
    "X3DHServerRound",
]


class X3DHServerRound:
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

    def run(
        self,
        clients: Optional[Set[str]] = None,
    ) -> None:
        """Run the X3DH setup across the network of peers.

        This method merely wraps the `async_run` one for
        synchronous execution, using an asyncio event loop.
        """
        asyncio.run(self.async_run(clients))

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
        messages = await self.netwk.wait_for_messages(clients)
        await self._transmit_x3dh_requests(requests, messages)
        # Gather X3DH responses and distribute them to their recipients.
        messages = await self.netwk.wait_for_messages(clients)
        await self._transmit_x3dh_responses(requests, messages)  # type: ignore
        # Verify that each and every client is done and okay.
        messages = await self.netwk.wait_for_messages(clients)
        await self._verify_finalized_okay(messages)

    async def _send_initial_requests(
        self,
        requests: Dict[str, List[str]],
    ) -> None:
        """Send an initial X3DH setup instruction to clients."""
        messages = {
            name: messaging.GenericMessage(
                action="x3dh-init", params={"n_reqs": len(peers)}
            )
            for name, peers in requests.items()
        }  # type: Dict[str, messaging.Message]
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
        requests = {name: [] for name in clients}  # type: Dict[str, List[str]]
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
        messages: Dict[str, messaging.Message],
    ) -> None:
        """Transmit X3DH requests from clients to their peers."""
        # Receive requests from clients and dispatch them by recipient.
        cli_reqs = {c: [] for c in messages}  # type: Dict[str, List[int]]
        for client, cli_msg in messages.items():
            assert isinstance(cli_msg, messaging.GenericMessage)
            assert cli_msg.action == "x3dh-request"
            assert len(cli_msg.params["requests"]) == len(requests[client])
            for cli, req in zip(requests[client], cli_msg.params["requests"]):
                cli_reqs[cli].append(req)
        # Send requests adressed to them to each and every client.
        messages = {
            client: messaging.GenericMessage(
                action="x3dh-request", params={"requests": requests}
            )
            for client, requests in cli_reqs.items()
        }
        await self.netwk.send_messages(messages)

    async def _transmit_x3dh_responses(
        self,
        requests: Dict[str, List[str]],
        messages: Dict[str, messaging.GenericMessage],
    ) -> None:
        """Transmit X3DH responses from clients to their peers."""
        for msg in messages.values():
            assert isinstance(msg, messaging.GenericMessage)
            assert msg.action == "x3dh-response"
        # For each client, fetch responses adressed to them.
        cli_resp = {c: [] for c in messages}  # type: Dict[str, List[int]]
        cli_indx = {c: 0 for c in messages}
        for dst, sources in requests.items():
            for src in sources:
                cli_resp[dst].append(
                    messages[src].params["responses"][cli_indx[src]]
                )
                cli_indx[src] += 1
        # Send responses adressed to them to each and every client.
        messages = {
            client: messaging.GenericMessage(
                action="x3dh-response", params={"responses": responses}
            )
            for client, responses in cli_resp.items()
        }
        await self.netwk.send_messages(messages)  # type: ignore

    async def _verify_finalized_okay(
        self,
        messages: Dict[str, messaging.Message],
    ) -> None:
        """Await confirmation from all clients that X3DH setup went fine."""
        for msg in messages.values():
            assert isinstance(msg, messaging.GenericMessage)
            assert msg.action == "x3dh-over-okay"


class X3DHClientRound:
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

    def run(
        self,
        msg: messaging.GenericMessage,
    ) -> None:
        """Run the X3DH setup across the network of peers.

        This method merely wraps the `async_run` one for
        synchronous execution, using an asyncio event loop.
        """
        asyncio.run(self.async_run(msg))

    async def async_run(
        self,
        msg: messaging.GenericMessage,
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
        msg = await self.netwk.check_message()  # type: ignore
        await self._respond_x3dh_requests(msg)
        # Process X3DH responses from peers and send back final status.
        msg = await self.netwk.check_message()  # type: ignore
        await self._process_x3dh_responses(msg)

    async def _create_x3dh_requests(
        self,
        msg: messaging.GenericMessage,
    ) -> None:
        """Setup and send X3DH requests to the server."""
        assert isinstance(msg, messaging.GenericMessage)
        assert msg.action == "x3dh-init"
        requests = [
            self.x3dhm.create_handshake_request()
            for _ in range(msg.params["n_reqs"])
        ]
        message = messaging.GenericMessage(
            action="x3dh-request",
            params={"requests": requests},
        )
        await self.netwk.send_message(message)

    async def _respond_x3dh_requests(
        self,
        msg: messaging.GenericMessage,
    ) -> None:
        """Process X3DH requests received from the server."""
        assert isinstance(msg, messaging.GenericMessage)
        assert msg.action == "x3dh-request"
        responses = [
            self.x3dhm.process_handshake_request(request)
            for request in msg.params["requests"]
        ]
        message = messaging.GenericMessage(
            action="x3dh-response",
            params={"responses": responses},
        )
        await self.netwk.send_message(message)

    async def _process_x3dh_responses(
        self,
        msg: messaging.GenericMessage,
    ) -> None:
        """Process X3DH responses received from the server."""
        assert isinstance(msg, messaging.GenericMessage)
        assert msg.action == "x3dh-response"
        for response in msg.params["responses"]:
            self.x3dhm.process_handshake_response(response)
        message = messaging.GenericMessage(action="x3dh-over-okay", params={})
        await self.netwk.send_message(message)
