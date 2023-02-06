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

"""Functional test for declearn.communication classes.

The test implemented here spawns a NetworkServer endpoint as well as one
or multiple NetworkClient ones, then runs parallelly routines that have
the clients register, and both sides exchange dummy messages. As such,
it only verifies that messages passing works, and does not constitute a
proper (ensemble of) unit test(s) of the classes.

However, if this passes, it means that registration and basic
message passing work properly, using the following scenarios:
* gRPC or WebSockets protocol
* SSL-secured communications or not
* 1-client or 3-clients cases

Note that the tests are somewhat slow when collected by pytest,
and that they make use of the multiprocessing library to isolate
the server and individual clients - which is not required when
running the code manually, and might require using '--full-trace'
pytest option to debug in case a test fails.

Note: running code that uses `asyncio.gather` on concurrent coroutines
is unsuccessful with gRPC due to spawned clients sharing the same peer
context. This may be fixed by implementing proper authentication.
"""

import asyncio
from typing import Any, Callable, Dict, List, Tuple

import pytest

from declearn.communication import (
    build_client,
    build_server,
    list_available_protocols,
)
from declearn.communication.api import NetworkClient, NetworkServer
from declearn.communication.messaging import GenericMessage
from declearn.test_utils import run_as_processes


async def client_routine(
    client: NetworkClient,
) -> None:
    """Basic client testing routine."""
    print("Registering")
    await client.register({"foo": "bar"})
    print("Receiving")
    message = await client.check_message()
    print(message)
    print("Sending")
    await client.send_message(GenericMessage(action="maybe", params={}))
    print("Receiving")
    message = await client.check_message()
    print(message)
    print("Sending")
    await client.send_message(message)
    print("Done!")


async def server_routine(
    server: NetworkServer,
    nb_clients: int = 1,
) -> None:
    """Basic server testing routine."""
    data_info = await server.wait_for_clients(
        min_clients=nb_clients, max_clients=nb_clients, timeout=5
    )
    print(data_info)
    print("Sending")
    await server.broadcast_message(
        GenericMessage(action="train", params={"let's": "go"})
    )
    print("Receiving")
    messages = await server.wait_for_messages()
    print(messages)
    print("Sending")
    messages = {
        client: GenericMessage("hello", {"name": client})
        for client in server.client_names
    }
    await server.send_messages(messages)
    print("Receiving")
    messages = await server.wait_for_messages()
    print(messages)
    print("Closing")


@pytest.mark.parametrize("nb_clients", [1, 3], ids=["1_client", "3_clients"])
@pytest.mark.parametrize("use_ssl", [False, True], ids=["ssl", "unsafe"])
@pytest.mark.parametrize("protocol", list_available_protocols())
def test_routines(
    protocol: str,
    nb_clients: int,
    use_ssl: bool,
    ssl_cert: Dict[str, str],
) -> None:
    """Test that the defined server and client routines run properly."""
    run_test_routines(protocol, nb_clients, use_ssl, ssl_cert)


def run_test_routines(
    protocol: str,
    nb_clients: int,
    use_ssl: bool,
    ssl_cert: Dict[str, str],
) -> None:
    """Test that the defined server and client routines run properly."""
    # Set up (func, args) tuples that specify concurrent routines.
    args = (protocol, nb_clients, use_ssl, ssl_cert)
    routines = [_build_server_func(*args)]
    routines.extend(_build_client_funcs(*args))
    # Run the former using isolated processes.
    exitcodes = run_as_processes(*routines)
    # Assert that all processes terminated properly.
    assert all(code == 0 for code in exitcodes)


def _build_server_func(
    protocol: str,
    nb_clients: int,
    use_ssl: bool,
    ssl_cert: Dict[str, str],
) -> Tuple[Callable[..., None], Tuple[Any, ...]]:
    """Return arguments to spawn and use a NetworkServer in a process."""
    server_cfg = {
        "protocol": protocol,
        "host": "127.0.0.1",
        "port": 8765,
        "certificate": ssl_cert["server_cert"] if use_ssl else None,
        "private_key": ssl_cert["server_pkey"] if use_ssl else None,
    }  # type: Dict[str, Any]

    # Define a coroutine that spawns and runs a server.
    async def server_coroutine() -> None:
        """Spawn a client and run `server_routine` in its context."""
        nonlocal nb_clients, server_cfg
        async with build_server(**server_cfg) as server:
            await server_routine(server, nb_clients)

    # Define a routine that runs the former.
    def server_func() -> None:
        """Run `server_coroutine`."""
        asyncio.run(server_coroutine())

    # Return the former as a (func, arg) tuple.
    return (server_func, tuple())


def _build_client_funcs(
    protocol: str,
    nb_clients: int,
    use_ssl: bool,
    ssl_cert: Dict[str, str],
) -> List[Tuple[Callable[..., None], Tuple[Any, ...]]]:
    """Return arguments to spawn and use NetworkClient objects in processes."""
    certificate = ssl_cert["client_cert"] if use_ssl else None
    server_uri = "localhost:8765"
    if protocol == "websockets":
        server_uri = f"ws{'s' * use_ssl}://{server_uri}"

    # Define a coroutine that spawns and runs a client.
    async def client_coroutine(
        name: str,
    ) -> None:
        """Spawn a client and run `client_routine` in its context."""
        nonlocal certificate, protocol, server_uri
        args = (protocol, server_uri, name, certificate)
        async with build_client(*args) as client:
            await client_routine(client)

    # Define a routine that runs the former.
    def client_func(name: str) -> None:
        """Run `client_coroutine`."""
        asyncio.run(client_coroutine(name))

    # Return a list of (func, args) tuples.
    return [(client_func, (f"client_{idx}",)) for idx in range(nb_clients)]
