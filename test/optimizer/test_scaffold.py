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

"""Unit tests for Scaffold OptiModule subclasses."""

from unittest import mock

import pytest

from declearn.model.api import Vector
from declearn.optimizer.modules import (
    AuxVar,
    ScaffoldAuxVar,
    ScaffoldClientModule,
    ScaffoldServerModule,
)
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
    assert_dict_equal,
)


@pytest.fixture(name="mock_gradients")
def fixture_mock_gradients(framework: FrameworkType) -> Vector:
    """Framework-specific, fixed-rng-based-valued mock gradients Vector."""
    test_case = GradientsTestCase(framework)
    return test_case.mock_gradient


class TestScaffoldClient:
    """Unit tests for 'ScaffoldClientModule'."""

    def test_initial_values(self) -> None:
        """Test that at first, a Scaffold module has zero-valued states."""
        module = ScaffoldClientModule()
        assert module.state == 0.0
        assert module.sglob == 0.0
        assert module.delta == 0.0

    def test_initial_auxvar_collection(self) -> None:
        """Test that initial aux_var collection warns and returns None."""
        module = ScaffoldClientModule()
        with pytest.warns(RuntimeWarning):
            aux_var = module.collect_aux_var()
        assert aux_var is None

    def test_first_run(self, mock_gradients: Vector) -> None:
        """Test that the first run works properly."""
        module = ScaffoldClientModule()
        # Test that there is no correction due to zero-valued states.
        output = module.run(mock_gradients)
        assert output == mock_gradients

    def test_auxvar_collection(self, mock_gradients: Vector) -> None:
        """Test that auxiliary variables collection works as expected."""
        module = ScaffoldClientModule()
        module.run(mock_gradients)
        aux_var = module.collect_aux_var()
        assert isinstance(aux_var, ScaffoldAuxVar)
        assert aux_var.delta == mock_gradients
        assert aux_var.state is None
        assert aux_var.clients == {module.uuid}

    def test_auxvar_processing_wrong_type(self) -> None:
        """Test 'process_aux_var' with improper type inputs."""
        module = ScaffoldClientModule()
        aux_var = mock.create_autospec(AuxVar, instance=True)
        with pytest.raises(TypeError):
            module.process_aux_var(aux_var)  # type: ignore

    def test_auxvar_processing_wrong_field(self) -> None:
        """Test 'process_aux_var' with client-like 'ScaffoldAuxVar'."""
        module = ScaffoldClientModule()
        with pytest.raises(KeyError):
            module.process_aux_var(ScaffoldAuxVar(delta=0.0))

    def test_process_aux_var(self, mock_gradients: Vector) -> None:
        """Test 'process_aux_var' with valid inputs, and its consequences."""
        module = ScaffoldClientModule()
        # Test that auxiliary variables processing updates correction.
        assert module.delta == module.sglob == 0.0
        module.process_aux_var(ScaffoldAuxVar(state=mock_gradients))
        assert module.sglob == mock_gradients
        assert module.delta == 0.0 - mock_gradients

    def test_run_with_correction(self, mock_gradients: Vector) -> None:
        """Test that the correction term is correctly applied."""
        module = ScaffoldClientModule()
        module.delta = mock_gradients  # forcefully overload
        zeros = mock_gradients - mock_gradients
        assert module.run(mock_gradients) == zeros


class TestScaffoldServer:
    """Unit tests for 'ScaffoldServerModule'."""

    def test_initial_values(self) -> None:
        """Test that at first, a Scaffold module has zero-valued states."""
        module = ScaffoldServerModule()
        assert module.s_state == 0.0
        assert not module.clients

    def test_initial_auxvar_collection(self) -> None:
        """Test that initial aux_var collection returns expected values."""
        module = ScaffoldServerModule()
        aux_var = module.collect_aux_var()
        assert isinstance(aux_var, ScaffoldAuxVar)
        assert aux_var.state == 0.0
        assert aux_var.delta is None
        assert not aux_var.clients

    def test_auxvar_processing_wrong_type(self) -> None:
        """Test 'process_aux_var' with improper type inputs."""
        module = ScaffoldServerModule()
        aux_var = mock.create_autospec(AuxVar, instance=True)
        with pytest.raises(TypeError):
            module.process_aux_var(aux_var)  # type: ignore

    def test_auxvar_processing_wrong_field(self) -> None:
        """Test 'process_aux_var' with server-like 'ScaffoldAuxVar'."""
        module = ScaffoldServerModule()
        with pytest.raises(KeyError):
            module.process_aux_var(ScaffoldAuxVar(state=0.0))

    def test_process_aux_var(self, mock_gradients: Vector) -> None:
        """Test 'process_aux_var' with valid inputs, and its consequences."""
        module = ScaffoldServerModule()
        # Test when processing initial auxiliary variables.
        aux_var = ScaffoldAuxVar(
            delta=mock_gradients,
            clients={"uuid0", "uuid1"},
        )
        module.process_aux_var(aux_var)
        assert module.s_state == aux_var.delta
        assert module.clients == aux_var.clients
        # Test when processing a second set of auxiliary variables.
        aux_var = ScaffoldAuxVar(
            delta=mock_gradients,
            clients={"uuid0", "uuid2"},
        )
        module.process_aux_var(aux_var)
        assert module.s_state == aux_var.delta + aux_var.delta  # type: ignore
        assert module.clients == {"uuid0", "uuid1", "uuid2"}

    def test_run(self, mock_gradients: Vector) -> None:
        """Test that 'run' leaves inputs as-is."""
        module = ScaffoldServerModule()
        module.s_state = mock_gradients
        vector = mock.create_autospec(Vector, instance=True)
        assert module.run(vector) is vector

    def test_collect_aux_var_initialized(self, mock_gradients: Vector) -> None:
        """Test 'collect_aux_var' after the module is initialized."""
        module = ScaffoldServerModule()
        module.s_state = mock_gradients
        module.clients = {"uuid0", "uuid1"}
        aux_var = module.collect_aux_var()
        assert isinstance(aux_var, ScaffoldAuxVar)
        assert aux_var.state == mock_gradients / 2
        assert aux_var.delta is None
        assert not aux_var.clients


def test_scaffold_routine(mock_gradients: Vector) -> None:
    """Conduct a mock client/server SCAFFOLD training routine.

    This test does not verify computations' correctness, but
    rather the formal coherence of client/server exchanges.
    """
    # Instantiate 10 clients and a server.
    clients = {f"client_{i}": ScaffoldClientModule() for i in range(10)}
    server = ScaffoldServerModule()
    # Run two training rounds.
    for rstep in range(2):
        # Emit and communicate initial states from the server to clients.
        aux_var = server.collect_aux_var()
        for client in clients:  # noqa: PLC0206
            clients[client].process_aux_var(aux_var)
        # Sample 5 participating clients. Have then run 3 training steps.
        participants = [f"client_{i}" for i in range(rstep, 10, 2)]
        for client in participants:
            for _ in range(3):
                clients[client].run(mock_gradients)
        # Emit, aggregate and process state updates from clients to server.
        clients_aux_var = [
            clients[client].collect_aux_var() for client in participants
        ]
        assert all(isinstance(x, ScaffoldAuxVar) for x in clients_aux_var)
        aux_var = sum(clients_aux_var)  # type: ignore
        server.process_aux_var(aux_var)


def test_scaffold_secagg_compatibility(mock_gradients: Vector) -> None:
    """Test that the Scaffold implementation is compatible with SecAgg."""
    # Instantiate a couple of clients, run an update and collect AuxVar.
    cli_a = ScaffoldClientModule()
    cli_b = ScaffoldClientModule()
    cli_a.run(mock_gradients * 0.5)
    cli_b.run(mock_gradients * 2.0)
    aux_a = cli_a.collect_aux_var()
    aux_b = cli_b.collect_aux_var()
    assert (aux_a is not None) and (aux_b is not None)
    # Extract fields as if for SecAgg and verify type correctness.
    sec_a, clr_a = aux_a.prepare_for_secagg()
    sec_b, clr_b = aux_b.prepare_for_secagg()
    assert isinstance(sec_a, dict) and isinstance(sec_b, dict)
    assert isinstance(clr_a, dict) and isinstance(clr_b, dict)
    # Conduct their aggregation as defined for SecAgg (but in cleartext).
    secagg = {key: val + sec_b[key] for key, val in sec_a.items()}
    clrtxt = {
        key: getattr(aux_a, f"aggregate_{key}", aux_a.default_aggregate)(
            val, clr_b[key]
        )
        for key, val in clr_a.items()
    }
    output = ScaffoldAuxVar(**secagg, **clrtxt)
    # Verify that results match expectations.
    expect = aux_a + aux_b
    assert_dict_equal(output.to_dict(), expect.to_dict())
