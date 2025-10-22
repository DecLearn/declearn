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

"""Unit tests for X3DH setup routines."""

import asyncio
import itertools
import secrets
from typing import Dict, List, Literal, Mapping, Optional

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from declearn.messaging import (
    Error,
    GenericMessage,
    Message,
    SerializedMessage,
)
from declearn.secagg.x3dh import run_x3dh_setup_client, run_x3dh_setup_server
from declearn.secagg.x3dh.messages import X3DHRequests, X3DHResponses
from declearn.test_utils import MockNetworkClient, MockNetworkServer


@pytest.fixture(name="id_keys", scope="module")
def id_keys_fixture() -> List[Ed25519PrivateKey]:
    """Provide with random Ed25519 keys generated once for this test file."""
    return [Ed25519PrivateKey.generate() for _ in range(5)]


async def run_server_routine(
    n_clients: int,
) -> None:
    """Prepare for and run the server-side setup routine."""
    async with MockNetworkServer() as netwk:
        await netwk.wait_for_clients(n_clients)
        await run_x3dh_setup_server(netwk)


async def run_client_routine(
    name: str,
    prv_key: Ed25519PrivateKey,
    trusted: List[Ed25519PublicKey],
) -> Dict[bytes, bytes]:
    """Prepare for and run the client-side setup routine."""
    async with MockNetworkClient(name=name) as netwk:
        await netwk.register({})
        s_keys = await run_x3dh_setup_client(netwk, prv_key, trusted)
    return s_keys


@pytest.mark.parametrize("n_clients", [2, 5])
@pytest.mark.asyncio
async def test_x3dh_setup_routines(
    n_clients: int,
    id_keys: List[Ed25519PrivateKey],
) -> None:
    """Test that the X3DH setup routines work properly."""
    # Setup the server and client routines.
    trusted = [key.public_key() for key in id_keys[:n_clients]]
    client_routines = [
        run_client_routine(f"client_{i}", id_keys[i], trusted)
        for i in range(n_clients)
    ]
    server_routine = run_server_routine(n_clients)
    # Run the routines concurrently and gather resulting objects.
    _, *s_keys = await asyncio.gather(server_routine, *client_routines)
    # Verify that outputs have expected types.
    for skd in s_keys:
        assert isinstance(skd, dict)
        assert all(isinstance(key, bytes) for key in skd)
        assert all(isinstance(val, bytes) for val in skd.values())
    # Verify that keys exist and differ for all pairs of clients.
    sec_keys: Dict[bytes, Dict[bytes, bytes]] = {
        idk.public_key().public_bytes_raw(): skd  # type: ignore
        for idk, skd in zip(id_keys[:n_clients], s_keys, strict=False)
    }
    idk_vals = [key.public_bytes_raw() for key in trusted]
    assert set(sec_keys) == set(idk_vals)
    assert {idk for sec in sec_keys.values() for idk in sec} == set(idk_vals)
    n_keys = len(set(key for sec in sec_keys.values() for key in sec.values()))
    assert n_keys == (n_clients * (n_clients - 1)) / 2
    # Verify that pairwise keys are identical, and 32-bytes long.
    for id_a, id_b in itertools.combinations_with_replacement(idk_vals, 2):
        if id_a == id_b:
            continue
        assert sec_keys[id_a][id_b] == sec_keys[id_b][id_a]
        assert len(sec_keys[id_a][id_b]) == 32


async def run_client_routine_sending_error(
    name: str,
    _,
) -> None:
    """Run faulty client code, sending an Error message out of the blue."""
    async with MockNetworkClient(name=name) as netwk:
        await netwk.register({})
        await netwk.recv_message()  # receive x3dh-init request
        await netwk.send_message(Error("test-error"))


async def run_client_routine_sending_wrong_type(
    name: str,
    _,
) -> SerializedMessage:
    """Run faulty client code, sending a message with wrong type."""
    async with MockNetworkClient(name=name) as netwk:
        await netwk.register({})
        await netwk.recv_message()  # receive x3dh-init request
        await netwk.send_message(GenericMessage(action="stub", params={}))
        return await netwk.recv_message()


async def run_client_routine_raising_x3dh_error(
    name: str,
    id_key: Ed25519PrivateKey,
) -> None:
    """Run faulty client code, sending an Error due to X3DH failure."""
    async with MockNetworkClient(name=name) as netwk:
        await netwk.register({})
        await run_x3dh_setup_client(netwk, prv_key=id_key, trusted=[])


FAULTY_CLIENT_ROUTINES = {
    "send_error": run_client_routine_sending_error,
    "wrong_type": run_client_routine_sending_wrong_type,
    "x3dh_error": run_client_routine_raising_x3dh_error,
}


@pytest.mark.parametrize("fault", list(FAULTY_CLIENT_ROUTINES))
@pytest.mark.asyncio
async def test_x3dh_setup_routines_with_faulty_client(
    id_keys: List[Ed25519PrivateKey],
    fault: Literal["send_error", "wrong_type", "x3dh_error"],
) -> None:
    """Test exception raising when a client is faulty."""
    n_clients = 3
    # Setup the server and client routines, including a faulty client.
    trusted = [key.public_key() for key in id_keys[:n_clients]]
    client_routines = [
        run_client_routine(f"client_{i}", id_keys[i], trusted)
        for i in range(n_clients - 1)
    ]
    server_routine = run_server_routine(n_clients)
    faulty_routine = FAULTY_CLIENT_ROUTINES[fault](
        f"client_{n_clients - 1}", id_keys[n_clients - 1]
    )
    # Run the routines concurrently and gather outputs or exceptions.
    server_exc, *clients_exc, faulty_out = await asyncio.gather(
        server_routine,
        *client_routines,
        faulty_routine,
        return_exceptions=True,
    )
    # Verify that expected exceptions were raised.
    assert isinstance(server_exc, RuntimeError)
    assert all(isinstance(exc, RuntimeError) for exc in clients_exc)
    if fault == "x3dh_error":
        assert isinstance(faulty_out, KeyError)
    elif fault == "send_error":
        assert faulty_out is None
    else:
        assert isinstance(faulty_out, SerializedMessage)
        assert faulty_out.message_cls is Error


async def run_server_routine_sending_error(
    n_clients: int,
) -> None:
    """Run faulty server code, sending an Error message."""
    async with MockNetworkServer() as netwk:
        await netwk.wait_for_clients(n_clients)
        await netwk.broadcast_message(Error("test-error"))


async def run_server_routine_sending_wrong_type(
    n_clients: int,
) -> None:
    """Run faulty server code, sending an Empty message."""
    async with MockNetworkServer() as netwk:
        await netwk.wait_for_clients(n_clients)
        await netwk.broadcast_message(GenericMessage(action="stub", params={}))


async def run_server_routine_tampering_with_requests(
    n_clients: int,
) -> None:
    """Run faulty server code, triggering X3DH failure on requests handling."""

    class TemperingNetworkServer(
        MockNetworkServer,
        register=False,  # type: ignore[call-arg]  # false-positive
    ):
        """Ad hoc NetworkServer subclass tempering with X3DH requests."""

        async def send_messages(
            self,
            messages: Mapping[str, Message],
            timeout: Optional[float] = None,
        ) -> None:
            # kwargs for readability; pylint: disable=arguments-differ
            for msg in messages.values():
                if isinstance(msg, X3DHRequests):
                    msg.requests = [
                        secrets.randbits(val.bit_length())
                        for val in msg.requests
                    ]
            await super().send_messages(messages, timeout)

    async with TemperingNetworkServer() as netwk:
        await netwk.wait_for_clients(n_clients)
        await run_x3dh_setup_server(netwk)


async def run_server_routine_tampering_with_responses(
    n_clients: int,
) -> None:
    """Run faulty server code, triggering X3DH failure on responses parsing."""

    class TemperingNetworkServer(
        MockNetworkServer,
        register=False,  # type: ignore[call-arg]  # false-positive
    ):
        """Ad hoc NetworkServer subclass tempering with X3DH responses."""

        async def send_messages(
            self,
            messages: Mapping[str, Message],
            timeout: Optional[float] = None,
        ) -> None:
            # kwargs for readability; pylint: disable=arguments-differ
            for msg in messages.values():
                if isinstance(msg, X3DHResponses):
                    msg.responses = [
                        secrets.randbits(val.bit_length())
                        for val in msg.responses
                    ]
            await super().send_messages(messages, timeout)

    async with TemperingNetworkServer() as netwk:
        await netwk.wait_for_clients(n_clients)
        await run_x3dh_setup_server(netwk)


FAULTY_SERVER_ROUTINES = {
    "send_error": run_server_routine_sending_error,
    "wrong_type": run_server_routine_sending_wrong_type,
    "x3dh_error_requests": run_server_routine_tampering_with_requests,
    "x3dh_error_responses": run_server_routine_tampering_with_responses,
}


@pytest.mark.parametrize("fault", list(FAULTY_SERVER_ROUTINES))
@pytest.mark.asyncio
async def test_x3dh_setup_routines_with_faulty_server(
    id_keys: List[Ed25519PrivateKey],
    fault: Literal["send_error", "wrong_type", "x3dh_error"],
) -> None:
    """Test exception raising when a client is faulty."""
    n_clients = 3
    # Setup the faulty server and proper client routines.
    trusted = [key.public_key() for key in id_keys[:n_clients]]
    client_routines = [
        run_client_routine(f"client_{i}", id_keys[i], trusted)
        for i in range(n_clients)
    ]
    server_routine = FAULTY_SERVER_ROUTINES[fault](n_clients)
    # Run the routines concurrently and gather outputs or exceptions.
    server_out, *clients_exc = await asyncio.gather(
        server_routine, *client_routines, return_exceptions=True
    )
    # Verify that expected exceptions were raised.
    if fault.startswith("x3dh_error"):
        assert isinstance(server_out, RuntimeError)
        expected = KeyError if fault.endswith("responses") else ValueError
        assert all(
            isinstance(exc, (expected, RuntimeError)) for exc in clients_exc
        )
    else:
        assert server_out is None
        assert all(isinstance(exc, RuntimeError) for exc in clients_exc)
