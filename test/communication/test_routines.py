# coding: utf-8

"""Functional test for declearn.communication classes.

The test implemented here spawns a Server endpoint as well as one
or multiple Client ones, then runs parallelly routines that have
the clients register, and both sides exchange dummy messages. As
such, it only verifies that messages passing works, and does not
constitute a proper (ensemble of) unit test(s) of the classes.

However, if this passes, it means that registration and basic
message passing work properly, using the following scenarios:
* gRPC or WebSockets protocol
* SSL-secured communications or not
* 1-client or 3-clients cases

Note that the tests are somewhat slow when collected by pytest,
and that they make use of the multiprocessing library to isolate
the server and individual clients - which is not required when
running the code manually, and might require using '--full-trace'
pytest option to debug in case a test fails. For unclear reasons,
running code that uses `asyncio.gather` on concurrent coroutines
was unsuccessful with pytest (resulting in slow 1-client tests,
and seemingly forever-running multiple-clients tests).
"""

import asyncio
import multiprocessing as mp
from typing import Dict, List

import pytest
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn.communication import build_client, build_server
from declearn.communication.api import Client, Server
from declearn.communication.messaging import GenericMessage


async def client_routine(
        client: Client,
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
        server: Server,
        nb_clients: int = 1,
    ) -> None:
    """Basic server testing routine."""
    data_info = await server.wait_for_clients(nb_clients)
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
@pytest.mark.parametrize("protocol", ["grpc", "websockets"])
def test_routines(
        protocol: Literal['grpc', 'websockets'],
        nb_clients: int,
        use_ssl: bool,
        ssl_cert: Dict[str, str],
    ) -> None:
    """Test that the defined server and client routines run properly."""
    run_test_routines(protocol, nb_clients, use_ssl, ssl_cert)


def run_test_routines(
        protocol: Literal['grpc', 'websockets'],
        nb_clients: int,
        use_ssl: bool,
        ssl_cert: Dict[str, str],
    ) -> None:
    """Test that the defined server and client routines run properly."""
    # Set up processes that isolately run a server and its clients
    args = (protocol, nb_clients, use_ssl, ssl_cert)
    processes = [_build_server_process(*args)]
    processes.extend(_build_client_processes(*args))
    # Start all processes.
    for process in processes:
        process.start()
    # Force termination in case any process raises an exception.
    while any(p.is_alive() for p in processes):
        if any(p.exitcode for p in processes):
            break
        processes[0].join(timeout=1)
    # Ensure all processes are terminated before exiting this function.
    for process in processes:
        process.terminate()
    # Assert that all processes terminated properly.
    assert all(process.exitcode == 0 for process in processes)


def _build_server_process(
        protocol: Literal['grpc', 'websockets'],
        nb_clients: int,
        use_ssl: bool,
        ssl_cert: Dict[str, str],
    ) -> mp.Process:
    """Set up and return a mp.Process that spawns and uses a Server."""
    server_cfg = {
        "protocol": protocol, "host": "localhost", "port": 8765,
        "certificate": ssl_cert["server_cert"] if use_ssl else None,
        "private_key": ssl_cert["server_pkey"] if use_ssl else None,
    }
    # Define a coroutine that spawns and runs a server.
    async def server_coroutine() -> None:
        """Spawn a client and run `server_routine` in its context."""
        nonlocal nb_clients, server_cfg
        async with build_server(**server_cfg) as server:  # type: ignore
            await server_routine(server, nb_clients)
    # Define a routine that runs the former.
    def server_process() -> None:
        """Run `server_coroutine`."""
        asyncio.run(server_coroutine())
    # Wrap the former in a Process and return it.
    return mp.Process(target=server_process)


def _build_client_processes(
        protocol: Literal['grpc', 'websockets'],
        nb_clients: int,
        use_ssl: bool,
        ssl_cert: Dict[str, str],
    ) -> List[mp.Process]:
    """Set up and return mp.Process that spawn and use Client objects."""
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
        await asyncio.sleep(1)
        async with build_client(*args) as client:
            await client_routine(client)
    # Define a routine that runs the former.
    def client_process(name: str) -> None:
        """Run `client_coroutine`."""
        asyncio.run(client_coroutine(name))
    # Wrap the former into Process objects and return them.
    return [
        mp.Process(target=client_process, args=(f"client_{idx}",))
        for idx in range(nb_clients)
    ]
