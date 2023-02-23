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

"""Shared code to define unit tests for declearn optimizer plug-in classes."""

import warnings
from typing import List, Tuple, Type, Union

import numpy as np

from declearn.model.api import Vector
from declearn.model.sklearn import NumpyVector
from declearn.optimizer.modules import OptiModule
from declearn.optimizer.regularizers import Regularizer
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
    list_available_frameworks,
)


Plugin = Union[OptiModule, Regularizer]


class PluginTestBase:
    """Shared unit tests for declearn optimizer plug-in classes.

    These tests verify that API-defined methods can be run and
    have the expected behaviour from the API's point of view -
    in other words, algorithmic correctness is *not* tested as
    it requires plugin-specific testing code.

    However, these tests assert that the plugins' `run` method
    effectively support gradients from a variety of frameworks
    (NumPy, TensorFlow, PyTorch) and that the outputs have the
    same values (up to reasonable numerical precision) for all
    of these.
    """

    @staticmethod
    def assert_equivalent(reg_a: Plugin, reg_b: Plugin) -> None:
        """Assert that two plug-ins are of same type and configuration."""
        assert type(reg_a) is type(reg_b)
        assert reg_a.get_config() == reg_b.get_config()

    def test_config(self, cls: Type[Plugin]) -> None:
        """Test a plug-in class's (get|from)_config methods."""
        plugin = cls()
        cfg = plugin.get_config()
        self.assert_equivalent(plugin, cls.from_config(cfg))

    def test_specs(self, cls: Type[Plugin]) -> None:
        """Test that a plug-in can be rebuilt using `from_specs`."""
        plugin = cls()
        base = (
            OptiModule if isinstance(plugin, OptiModule) else Regularizer
        )  # type: Type[Union[OptiModule, Regularizer]]
        name = plugin.name
        config = plugin.get_config()
        self.assert_equivalent(plugin, base.from_specs(name, config))

    @staticmethod
    def _run_plugin(
        plugin: Plugin,
        test_case: GradientsTestCase,
    ) -> Tuple[Vector, Vector]:
        """Generate random inputs and run them through a given plug-in.

        Return the input and output "gradient" vectors.
        """
        inputs = test_case.mock_gradient
        if isinstance(plugin, OptiModule):
            output = plugin.run(inputs)
        else:
            params = test_case.mock_gradient  # model params (same specs)
            output = plugin.run(inputs, params)
        return inputs, output

    def test_run(self, cls: Type[Plugin], framework: FrameworkType) -> None:
        """Test a plug-in's run method using a given framework.

        Note: Only check that input and output gradients have
              same specs, not their algorithmic correctness.
        """
        test_case = GradientsTestCase(framework)
        plugin = cls()
        inputs, output = self._run_plugin(plugin, test_case)
        assert isinstance(output, test_case.vector_cls)
        assert output.coefs.keys() == inputs.coefs.keys()
        assert output.shapes() == inputs.shapes()
        assert output.dtypes() == inputs.dtypes()

    def test_run_equivalence(self, cls: Type[Plugin]) -> None:
        """Test that a plug-in's run is equivalent for all frameworks.

        Note: If a framework's run fails, warn about it but keep going,
              as `test_run` is sufficient to report framework failure.
              If no framework was correctly run, fail this test.
        """
        # Collect outputs from a newly-created plugin for each framework.
        results = []  # type: List[NumpyVector]
        for fwk in list_available_frameworks():
            f_case = GradientsTestCase(fwk)  # type: ignore
            plugin = cls()
            try:
                _, output = self._run_plugin(plugin, f_case)
            except Exception:  # pylint: disable=broad-except
                warnings.warn(
                    f"Skipping framework '{fwk}' in equivalence test."
                )
            else:
                coefs = {
                    key: f_case.to_numpy(val)
                    for key, val in output.coefs.items()
                }
                results.append(NumpyVector(coefs))
        # Check that all collected results have the same values.
        assert results
        assert all(
            np.allclose(results[0].coefs[key], val)  # for numerical precision
            for res in results[1:]
            for key, val in res.coefs.items()
        )
