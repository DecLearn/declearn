# coding: utf-8

"""Unit tests for Scaffold OptiModule subclasses."""

import sys

import pytest

from declearn2.model.api import Vector
from declearn2.optimizer.modules import (
    ScaffoldClientModule, ScaffoldServerModule
)
# dirty trick to import from `test_modules.py`;
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append('.')
from test_modules import FRAMEWORKS, Framework, GradientsTestCase


@pytest.fixture(name="mock_gradients")
def fixture_mock_gradients(framework: Framework) -> Vector:
    """Framework-specific, fixed-rng-based-valued mock gradients Vector."""
    test_case = GradientsTestCase(framework)
    return test_case.mock_gradient


@pytest.mark.parametrize('framework', FRAMEWORKS)
def test_scaffold_client(mock_gradients: Vector) -> None:
    """Conduct a series of co-dependent unit tests on ScaffoldClientModule."""
    module = ScaffoldClientModule()
    assert module.delta == 0.
    # Test that initial aux_var collection fails.
    with pytest.raises(RuntimeError):
        module.collect_aux_var()
    # Test run correctness (no correction at state 0).
    output = module.run(mock_gradients)
    assert output == mock_gradients
    # Test aux_var collection after run.
    aux_var = module.collect_aux_var()
    assert aux_var == {"state": mock_gradients}
    # Test mock aux_var processing (as though server-emitted).
    with pytest.raises(KeyError):
        module.process_aux_var({})
    module.process_aux_var({"delta": mock_gradients})
    assert module.delta == mock_gradients
    # Test run correctness (with correction).
    zeros = (mock_gradients - mock_gradients)
    assert module.run(mock_gradients) == zeros


@pytest.mark.parametrize('framework', FRAMEWORKS)
def test_scaffold_server(mock_gradients: Vector) -> None:
    """Conduct a series of co-dependent unit tests on ScaffoldServerModule."""
    module = ScaffoldServerModule()
    assert module.state == 0.
    # Test initial aux_var collection.
    aux_var = module.collect_aux_var()
    assert not aux_var
    # Test mock aux_var processing (as though clients-emitted).
    with pytest.raises(KeyError):
        module.process_aux_var({"client": {"lorem": "ipsum"}})
    with pytest.raises(TypeError):
        module.process_aux_var({"client": {"state": [0.]}})
    module.process_aux_var(
        {str(i): {"state": mock_gradients} for i in range(5)}
    )
    assert module.s_loc == {str(i): mock_gradients for i in range(5)}
    assert isinstance(module.state, type(mock_gradients))
    # Take numerical precision issues into account when checking values.
    mock_unprecise = (5 * mock_gradients / 5)
    assert module.state == mock_unprecise
    # Test run correctness (no correction as per algorithm).
    assert module.run(mock_gradients) == mock_gradients
    # Test aux_var collection after a round.
    zeros = (mock_gradients - mock_unprecise)
    aux_var = module.collect_aux_var()
    assert aux_var == {str(i): {"delta": zeros} for i in range(5)}


@pytest.mark.parametrize(
    'client_aware', [True, False], ids=['ClientAware', 'ClientBlind']
)
@pytest.mark.parametrize('framework', FRAMEWORKS)
def test_scaffold_routine(client_aware: bool, mock_gradients: Vector) -> None:
    """Conduct a mock client/server SCAFFOLD training routine.

    This test does not verify computations' correctness, but
    rather the formal coherence of client/server exchanges.

    Based on `client_aware`, the server may or not know the
    list of clients in advance.
    """
    # Instantiate 10 clients and a server.
    clients = {
        f"client_{i}": ScaffoldClientModule() for i in range(10)
    }
    server = ScaffoldServerModule(list(clients) if client_aware else None)
    # Run two training rounds.
    for rstep in range(2):
        # Emit and communicate initial states from server to clients.
        shared = server.collect_aux_var()
        assert shared is not None
        for client, aux_var in shared.items():
            clients[client].process_aux_var(aux_var)
        # Sample 5 participating clients. Have then run 3 training steps.
        participants = [f"client_{i}" for i in range(rstep, 10, 2)]
        for client in participants:
            for _ in range(3):
                clients[client].run(mock_gradients)
        # Emit, communicate and process state updates from clients to server.
        shared = {
            client: clients[client].collect_aux_var()
            for client in participants
        }
        server.process_aux_var(shared)
