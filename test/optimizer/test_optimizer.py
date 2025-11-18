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
# type: ignore  # mock objects everywhere

"""Unit tests for `declearn.optimizer.Optimizer`."""

from typing import Any, Dict, Tuple
from unittest import mock
from uuid import uuid4

import pytest

from declearn.model.api import Model, Vector
from declearn.optimizer import Optimizer
from declearn.optimizer.modules import AuxVar, OptiModule
from declearn.optimizer.regularizers import Regularizer
from declearn.optimizer.schedulers import Scheduler
from declearn.test_utils import assert_json_serializable_dict


class MockOptiModule(OptiModule):
    """Type-registered mock OptiModule subclass."""

    name = f"mock-{uuid4()}"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs

    def run(self, gradients: Vector) -> Vector:
        return gradients

    def get_config(self) -> Dict[str, Any]:
        return self.kwargs


class MockRegularizer(Regularizer):
    """Type-registered mock Regularizer subclass."""

    name = f"mock-{uuid4()}"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs

    def run(self, gradients: Vector, weights: Vector) -> Vector:
        return gradients

    def get_config(self) -> Dict[str, Any]:
        return self.kwargs


class MockScheduler(Scheduler):
    """Type-registered mock Scheduler subclass."""

    name = f"mock-{uuid4()}"

    def compute_value(
        self,
        step: int,
        round_: int,
    ) -> float:
        return self.base


class TestOptimizer:
    """Unit tests for `declearn.optimizer.Optimizer`."""

    # test-grouping class; pylint: disable=too-many-public-methods

    def test_init_vanilla(self) -> None:
        """Test `Optimizer` instantiation without plug-ins."""
        optimizer = Optimizer(lrate=0.001)
        assert optimizer.lrate == 0.001
        assert optimizer.w_decay == 0.0
        assert isinstance(optimizer.regularizers, list)
        assert not optimizer.regularizers
        assert isinstance(optimizer.modules, list)
        assert not optimizer.modules

    def test_init_with_lrate_scheduler_instance(self) -> None:
        """Test `Optimizer` instantiation with a learning rate Scheduler."""
        optimizer = Optimizer(lrate=MockScheduler(base=0.001))
        assert optimizer.lrate == 0.001
        assert optimizer.w_decay == 0.0

    def test_init_with_decay_scheduler_instance(self) -> None:
        """Test `Optimizer` instantiation with a weight decay Scheduler."""
        optimizer = Optimizer(lrate=0.1, w_decay=MockScheduler(base=0.001))
        assert optimizer.lrate == 0.1
        assert optimizer.w_decay == 0.001

    def test_init_with_scheduler_specs(self) -> None:
        """Test `Optimizer` instantiation with Scheduler specs."""
        optimizer = Optimizer(
            lrate=(MockScheduler.name, {"base": 0.001}),
            w_decay=(MockScheduler.name, {"base": 0.9}),
        )
        assert optimizer.lrate == 0.001
        assert optimizer.w_decay == 0.9

    def test_init_with_plugin_instances(self) -> None:
        """Test `Optimizer` instantiation with plug-in instances."""
        optimizer = Optimizer(
            lrate=0.001,
            regularizers=[MockRegularizer()],
            modules=[MockOptiModule()],
        )
        assert len(optimizer.regularizers) == 1
        assert isinstance(optimizer.regularizers[0], MockRegularizer)
        assert len(optimizer.modules) == 1
        assert isinstance(optimizer.modules[0], MockOptiModule)

    def test_init_with_plugin_names(self) -> None:
        """Test `Optimizer` instantiation with plug-in name identifiers."""
        optimizer = Optimizer(
            lrate=0.001,
            regularizers=[MockRegularizer.name],
            modules=[MockOptiModule.name],
        )
        assert len(optimizer.regularizers) == 1
        assert isinstance(optimizer.regularizers[0], MockRegularizer)
        assert len(optimizer.modules) == 1
        assert isinstance(optimizer.modules[0], MockOptiModule)

    def test_init_with_plugin_specs(self) -> None:
        """Test `Optimizer` instantiation with plug-in sepcs tuples."""
        kwargs = {"mock": "kwargs"}
        optimizer = Optimizer(
            lrate=0.001,
            regularizers=[(MockRegularizer.name, kwargs)],
            modules=[(MockOptiModule.name, kwargs)],
        )
        assert len(optimizer.regularizers) == 1
        assert isinstance(optimizer.regularizers[0], MockRegularizer)
        assert optimizer.regularizers[0].kwargs == kwargs
        assert len(optimizer.modules) == 1
        assert isinstance(optimizer.modules[0], MockOptiModule)
        assert optimizer.modules[0].kwargs == kwargs

    def test_init_errors(self) -> None:
        """Test `Optimizer` instantiation with wrongful plug-in specs."""
        with pytest.raises(TypeError):
            Optimizer(
                lrate=0.001,
                regularizers=[{MockRegularizer.name: {}}],
            )
        with pytest.raises(TypeError):
            Optimizer(
                lrate=0.001,
                modules=[{MockOptiModule.name: {}}],
            )

    def test_get_config(self) -> None:
        """Test that `Optimizer.get_config` collects a serializable dict."""
        optimizer = Optimizer(
            lrate=0.001,
            w_decay=0.005,
            regularizers=[MockRegularizer(arg="regularizer")],
            modules=[MockOptiModule(arg="optimodule")],
        )
        config = optimizer.get_config()
        assert_json_serializable_dict(config)

    def test_from_config(self) -> None:
        """Test that `Optimizer.from_config(opt.get_config())` works."""
        # Test that an Optimizer can be created from a first one's config.
        opti_a = Optimizer(
            lrate=0.001,
            w_decay=0.005,
            regularizers=[MockRegularizer(arg="regularizer")],
            modules=[MockOptiModule(arg="optimodule")],
        )
        config = opti_a.get_config()
        opti_b = Optimizer.from_config(config)
        assert isinstance(opti_b, Optimizer)
        # Test that the second Optimizer abides by the expected specs.
        assert opti_b.get_config() == config
        assert opti_b.lrate == 0.001
        assert opti_b.w_decay == 0.005
        assert len(opti_b.regularizers) == 1
        assert isinstance(opti_b.regularizers[0], MockRegularizer)
        assert opti_b.regularizers[0].kwargs == {"arg": "regularizer"}
        assert len(opti_b.modules) == 1
        assert isinstance(opti_b.modules[0], MockOptiModule)
        assert opti_b.modules[0].kwargs == {"arg": "optimodule"}

    def test_compute_updates_from_gradients(self) -> None:
        """Test, using mocks, that updates are computed with expected calls."""
        # Set up an Optimizer with mock attributes and run the computation.
        optim = Optimizer(
            lrate=mock.create_autospec(Scheduler, instance=True),
            w_decay=mock.create_autospec(Scheduler, instance=True),
            regularizers=[
                mock.create_autospec(Regularizer, instance=True)
                for _ in range(2)
            ],
            modules=[
                mock.create_autospec(OptiModule, instance=True)
                for _ in range(2)
            ],
        )
        model = mock.create_autospec(Model, instance=True)
        grads = mock.create_autospec(Vector, instance=True)
        optim.compute_updates_from_gradients(model, grads)
        # Check that model weights were fetched and get their mock value.
        model.get_weights.assert_called_once_with(trainable=True)
        weights = model.get_weights.return_value
        # Check that the inputs went through the expected plug-ins pipeline.
        output = grads  # initial inputs
        for reg in optim.regularizers:
            reg.run.assert_called_once_with(output, weights)
            output = reg.run.return_value
        for mod in optim.modules:
            mod.run.assert_called_once_with(output)
            output = mod.run.return_value
        # Check that the learning rate and weight-decay were properly used.
        optim.lrate.__mul__.assert_called_once_with(output)  # lrate * updates
        optim.w_decay.__mul__.assert_called_once_with(weights)  # w_d * weights

    def test_collect_aux_var(self) -> None:
        """Test, using mocks, that `Optimizer.collect_aux_var` works."""
        optim = Optimizer(
            lrate=0.001,
            modules=[mock.create_autospec(OptiModule)],
        )
        aux_var = optim.collect_aux_var()
        assert isinstance(aux_var, dict)
        for mod in optim.modules:
            mod.collect_aux_var.assert_called_once()

    def test_process_aux_var(self) -> None:
        """Test, using mocks, that `Optimizer.process_aux_var` works."""
        # Set up an Optimizer with a couple of mock modules.
        mod_a = mock.create_autospec(OptiModule, instance=True)
        mod_a.name = "mock-a"
        mod_a.aux_name = "mock"
        mod_b = mock.create_autospec(OptiModule, instance=True)
        mod_b.name = "mock-b"
        mod_b.aux_name = None
        optim = Optimizer(lrate=0.001, modules=[mod_a, mod_b])
        # Process "valid" auxiliary variables and verify their proper passing.
        aux_var = {"mock": mock.create_autospec(AuxVar, instance=True)}
        assert optim.process_aux_var(aux_var) is None
        mod_a.process_aux_var.assert_called_once_with(aux_var["mock"])
        mod_b.process_aux_var.assert_not_called()

    def test_process_aux_var_invalid(self) -> None:
        """Test that `Optimizer.process_aux_var` raises expected errors."""
        optim = Optimizer(lrate=0.001)
        aux_var = {"mock": mock.create_autospec(AuxVar, instance=True)}
        with pytest.raises(KeyError):
            optim.process_aux_var(aux_var)

    def test_start_round(self) -> None:
        """Test, using mocks, that `Optimizer.start_round` works."""
        lrate = mock.create_autospec(Scheduler, instance=True)
        decay = mock.create_autospec(Scheduler, instance=True)
        optim = Optimizer(
            lrate=lrate,
            w_decay=decay,
            regularizers=[mock.create_autospec(Regularizer, instance=True)],
        )
        assert optim.start_round() is None
        for reg in optim.regularizers:
            reg.on_round_start.assert_called_once()
        lrate.on_round_start.assert_called_once()
        decay.on_round_start.assert_called_once()

    def test_run_train_step(self) -> None:
        """Test, using mocks, that `Optimizer.run_train_step` works."""
        # Set up an Optimizer, and mock inputs to `run_train_step`.
        optim = Optimizer(lrate=0.001)
        model = mock.create_autospec(Model, instance=True)
        grads = mock.create_autospec(Vector, instance=True)
        model.compute_batch_gradients.return_value = grads
        batch = mock.Mock()
        # Call `run_train_step` and check that things flow as expected.
        with mock.patch.object(optim, "apply_gradients"):
            assert optim.run_train_step(model, batch, sclip=5.0) is None
            model.compute_batch_gradients.assert_called_once_with(
                batch=batch, max_norm=5.0
            )
            # mock-patched; pylint: disable=no-member
            optim.apply_gradients.assert_called_once_with(model, grads)

    def test_apply_gradients(self) -> None:
        """Test, using mocks, that `Optimizer.apply_gradients` works."""
        # Set up an Optimizer, and mock inputs to `apply_gradients`.
        optim = Optimizer(lrate=0.001)
        model = mock.create_autospec(Model, instance=True)
        grads = mock.create_autospec(Vector, instance=True)
        updts = mock.create_autospec(Vector, instance=True)
        # Call `run_train_step` and check that things flow as expected.
        with mock.patch.object(optim, "compute_updates_from_gradients"):
            optim.compute_updates_from_gradients.return_value = updts
            assert optim.apply_gradients(model, grads) is None
            # mock-patched; pylint: disable=no-member
            optim.compute_updates_from_gradients.assert_called_once_with(
                model, grads
            )
            model.apply_updates.assert_called_once_with(updts)

    def test_get_state(self) -> None:
        """Test that `Optimizer.get_state` collects state variables."""
        # Set up an Optimizer with a mock stateful module.
        module = mock.create_autospec(OptiModule, instance=True)
        optim = Optimizer(lrate=0.001, modules=[module])
        # Check that the states are properly collected.
        state = optim.get_state()
        assert isinstance(state, dict)
        assert state.keys() == {"modules", "lrate", "w_decay"}
        module.get_state.assert_called_once()

    def _setup_for_set_state(
        self,
    ) -> Tuple[OptiModule, Dict[str, Any], Optimizer]:
        """Shared setup for `set_state` unit tests."""
        module = mock.create_autospec(OptiModule, instance=True)
        module.name = "mock-module"
        states = {"state": mock.Mock()}
        module.get_state.return_value = states
        optim = Optimizer(lrate=0.001, modules=[module])
        return module, states, optim

    def test_set_state(self) -> None:
        """Test that `Optimizer.set_state` propagates state variables."""
        module, states, optim = self._setup_for_set_state()
        # Run state-access methods and check that states are properly set.
        assert optim.set_state(optim.get_state()) is None
        module.set_state.assert_called_once_with(states)

    def test_set_state_invalid_dict(self) -> None:
        """Test that `Optimizer.set_state` raises an expected exception.

        Case when missing the "modules" main key.
        """
        module, _, optim = self._setup_for_set_state()
        with pytest.raises(KeyError):
            optim.set_state({})
        module.get_state.assert_not_called()
        module.set_state.assert_not_called()

    def test_set_state_invalid_empty(self) -> None:
        """Test that `Optimizer.set_state` raises an expected exception.

        Case when missing states for the wrapped module.
        """
        module, _, optim = self._setup_for_set_state()
        with pytest.raises(KeyError):
            optim.set_state({"modules": []})
        module.get_state.assert_not_called()
        module.set_state.assert_not_called()

    def test_set_state_invalid_mislabeled(self) -> None:
        """Test that `Optimizer.set_state` raises an expected exception.

        Case when containing mislabeled states.
        """
        module, states, optim = self._setup_for_set_state()
        bad_state = {
            "modules": [("mislabeled", mock.Mock())],
            "lrate": {"steps": 0, "rounds": 0},
            "w_decay": {"steps": 0, "rounds": 0},
        }
        with pytest.raises(KeyError):
            optim.set_state(bad_state)
        module.get_state.assert_called_once()
        module.set_state.assert_called_once_with(states)  # reset

    def test_set_state_invalid_second(self) -> None:
        """Test that `Optimizer.set_state` raises an expected exception.

        Case when a first module's states are okay but a second's are bad.
        """
        # Set up an optimizer with a stateful mock module.
        mod_a, states, optim = self._setup_for_set_state()
        # Add a second, stateless mock module to the optimizer.
        mod_b = mock.create_autospec(OptiModule, instance=True)
        mod_b.name = "mock-bis"
        mod_b.get_state.return_value = {}
        optim.modules.append(mod_b)
        # Run the invalid `set_state` and test assertions.
        new_state = mock.Mock()
        bad_state = {
            "modules": [(mod_a.name, new_state), ("other", new_state)],
            "lrate": {"steps": 0, "rounds": 0},
            "w_decay": {"steps": 0, "rounds": 0},
        }
        with pytest.raises(KeyError):
            optim.set_state(bad_state)
        mod_a.get_state.assert_called_once()
        mod_b.get_state.assert_called_once()
        mod_b.set_state.assert_called_once_with({})  # reset initial state
        mod_a.set_state.assert_has_calls(
            # calls: assign new state, then reset due to the raised error
            [mock.call(new_state), mock.call(states)]
        )

    def test_set_state_flawed_module(self) -> None:
        """Test that `Optimizer.set_state` raises an expected exception.

        Case when a wrapped module has wrongful get/set states.
        """
        # Set up an optimizer with a stateful mock module.
        module, states, optim = self._setup_for_set_state()
        # Make the module's `set_state` method fail no matter the inputs.
        module.set_state.side_effect = KeyError("Wrong input states.")
        # Run `test_set`: expect RuntimeError due to failure to reset.
        new_state = {
            "modules": [(module.name, {})],
            "lrate": {"steps": 0, "rounds": 0},
            "w_decay": {"steps": 0, "rounds": 0},
        }
        with pytest.raises(RuntimeError):
            optim.set_state(new_state)
            module.set_state.assert_has_calls(
                # calls: assign new state, then reset due to the raised error
                [mock.call({}), mock.call(states)]
            )
