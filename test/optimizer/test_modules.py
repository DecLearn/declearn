# coding: utf-8

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
import json
import os
import sys
import tempfile
from typing import Any, Dict, Type

import pytest
from declearn.optimizer.modules import NoiseModule, OptiModule
from declearn.test_utils import FrameworkType, GradientsTestCase
from declearn.utils import json_pack, json_unpack
from declearn.utils._register import REGISTRIES

# relative import; pylint: disable=wrong-import-order, wrong-import-position
# fmt: off
sys.path.append(".")
from optim_testing import PluginTestBase
sys.path.pop()
# fmt: on

# unproper but efficient way to list modules; pylint: disable=protected-access
OPTIMODULE_SUBCLASSES = REGISTRIES["OptiModule"]._reg
# pylint: enable=protected-access


@pytest.mark.parametrize(
    "cls", OPTIMODULE_SUBCLASSES.values(), ids=OPTIMODULE_SUBCLASSES.keys()
)
class TestOptiModule(PluginTestBase):
    """Unit tests for declearn.optimizer.modules.OptiModule subclasses."""

    @staticmethod
    def assert_json_serializable(sdict: Dict[str, Any]) -> None:
        """Assert that an input is JSON-serializable using declearn hooks."""
        dump = json.dumps(sdict, default=json_pack)
        load = json.loads(dump, object_hook=json_unpack)
        assert isinstance(load, dict)
        assert load.keys() == sdict.keys()
        assert all(load[key] == sdict[key] for key in sdict)

    def test_serialization(self, cls: Type[OptiModule]) -> None:
        """Test an OptiModule's (de)?serialize methods."""
        module = cls()
        cfg = module.serialize()
        self.assert_equivalent(module, cls.deserialize(cfg))

    def test_serialization_json(self, cls: Type[OptiModule]) -> None:
        """Test an OptiModule's JSON-file deserialization."""
        module = cls()
        cfg = module.serialize()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "module.json")
            cfg.to_json(path)
            self.assert_equivalent(module, cls.deserialize(path))

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
            self.assert_json_serializable(aux_var)

    def test_get_state_initial(self, cls: Type[OptiModule]) -> None:
        """Test an OptiModule's get_state method at instanciation."""
        module = cls()
        states = module.get_state()
        assert isinstance(states, dict)
        self.assert_json_serializable(states)

    def test_get_state_updated(
        self, cls: Type[OptiModule], framework: FrameworkType
    ) -> None:
        """Test an OptiModule's get_state method after an update."""
        module = cls()
        if module.get_state():  # skip the test if the module is stateless
            test_case = GradientsTestCase(framework)
            module.run(test_case.mock_gradient)
            states = module.get_state()
            assert isinstance(states, dict)
            self.assert_json_serializable(states)

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
