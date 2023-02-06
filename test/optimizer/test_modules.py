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

"""Unit tests for OptiModule subclasses.

This script implements unit tests that are automatically run
for each and every OptiModule subclass type-registered under
the "OptiModule" group name.

These tests verify that API-defined methods can be run and
have the expected behaviour from the API's point of view -
in other words, algorithmic correctness is *not* tested as
it requires module-specific testing code.

However, these tests assert that the modules' `run` method
effectively support gradients from a variety of frameworks
(NumPy, TensorFlow, PyTorch) and that the outputs have the
same values (up to reasonable numerical precision) for all
of these.
"""

import functools
import sys
from typing import Type

import pytest
from declearn.optimizer.modules import NoiseModule, OptiModule
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
    assert_json_serializable_dict,
)
from declearn.utils import access_types_mapping

# relative import; pylint: disable=wrong-import-order, wrong-import-position
# fmt: off
sys.path.append(".")
from optim_testing import PluginTestBase
sys.path.pop()
# fmt: on


OPTIMODULE_SUBCLASSES = access_types_mapping(group="OptiModule")


@pytest.mark.parametrize(
    "cls", OPTIMODULE_SUBCLASSES.values(), ids=OPTIMODULE_SUBCLASSES.keys()
)
class TestOptiModule(PluginTestBase):
    """Unit tests for declearn.optimizer.modules.OptiModule subclasses."""

    def test_collect_aux_var(
        self, cls: Type[OptiModule], framework: FrameworkType
    ) -> None:
        """Test an OptiModule's collect_aux_var method."""
        test_case = GradientsTestCase(framework)
        module = cls()
        module.run(test_case.mock_gradient)
        aux_var = module.collect_aux_var()
        assert (aux_var is None) or isinstance(aux_var, dict)
        if isinstance(aux_var, dict):
            assert_json_serializable_dict(aux_var)

    def test_get_state_initial(self, cls: Type[OptiModule]) -> None:
        """Test an OptiModule's get_state method at instanciation."""
        module = cls()
        states = module.get_state()
        assert_json_serializable_dict(states)

    def test_get_state_updated(
        self, cls: Type[OptiModule], framework: FrameworkType
    ) -> None:
        """Test an OptiModule's get_state method after an update."""
        module = cls()
        if module.get_state():  # skip the test if the module is stateless
            test_case = GradientsTestCase(framework)
            module.run(test_case.mock_gradient)
            states = module.get_state()
            assert_json_serializable_dict(states)

    def test_set_state_initial(
        self, cls: Type[OptiModule], framework: FrameworkType
    ) -> None:
        """Test an OptiModule's set_state method to reset states."""
        module = cls()
        initial = module.get_state()
        if initial:  # skip the test if the module is stateless
            test_case = GradientsTestCase(framework)
            module.run(test_case.mock_gradient)
            module.set_state(initial)
            assert module.get_state() == initial

    def test_set_state_updated(
        self, cls: Type[OptiModule], framework: FrameworkType
    ) -> None:
        """Test an OptiModule's set_state method to fast-forward states."""
        module = cls()
        if module.get_state():  # skip the test if the module is stateless
            test_case = GradientsTestCase(framework)
            module.run(test_case.mock_gradient)
            states = module.get_state()
            module = cls()
            module.set_state(states)
            assert module.get_state() == states

    def test_set_state_failure(self, cls: Type[OptiModule]) -> None:
        """Test that an OptiModule's set_state raises an excepted error."""
        module = cls()
        states = module.get_state()
        if states:  # skip the test if the module is stateless
            with pytest.raises(KeyError):
                states.pop(list(states)[0])  # remove a state variable
                module.set_state(states)

    def test_run_equivalence(  # type: ignore
        self, cls: Type[OptiModule]
    ) -> None:
        # For Noise-addition mechanisms, seed the (unsafe) RNG.
        if issubclass(cls, NoiseModule):
            cls = functools.partial(
                cls, safe_mode=False, seed=0
            )  # type: ignore  # partial wraps the __init__ method
        # Run the unit test.
        super().test_run_equivalence(cls)
