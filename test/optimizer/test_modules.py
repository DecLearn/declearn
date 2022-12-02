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

import json
import os
import sys
import tempfile
from typing import Type

import pytest

from declearn.optimizer.modules import OptiModule
from declearn.utils import json_pack, json_unpack
from declearn.utils._register import REGISTRIES


# dirty trick to import from `model_testing.py`;
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(".")
from optim_testing import (
    FRAMEWORKS,
    Framework,
    GradientsTestCase,
    PluginTestBase,
)

# unproper but efficient way to list modules; pylint: disable=protected-access
OPTIMODULE_SUBCLASSES = REGISTRIES["OptiModule"]._reg
# pylint: enable=protected-access


@pytest.mark.parametrize(
    "cls", OPTIMODULE_SUBCLASSES.values(), ids=OPTIMODULE_SUBCLASSES.keys()
)
class TestOptiModule(PluginTestBase):
    """Unit tests for declearn.optimizer.modules.OptiModule subclasses."""

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

    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_collect_aux_var(
        self, cls: Type[OptiModule], framework: Framework
    ) -> None:
        """Test an OptiModule's collect_aux_var method."""
        test_case = GradientsTestCase(framework)
        module = cls()
        module.run(test_case.mock_gradient)
        aux_var = module.collect_aux_var()
        assert (aux_var is None) or isinstance(aux_var, dict)
        if isinstance(aux_var, dict):
            dump = json.dumps(aux_var, default=json_pack)
            assert isinstance(dump, str)
            assert json.loads(dump, object_hook=json_unpack) == aux_var
