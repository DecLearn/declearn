# coding: utf-8

"""Shared code to instantiate framework-specific vectors."""

import warnings
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import pytest

with warnings.catch_warnings():  # silence tensorflow import-time warnings
    warnings.simplefilter("ignore")
    import tensorflow as tf  # type: ignore
import torch
from numpy.typing import ArrayLike
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn.model.api import NumpyVector, Vector
from declearn.model.tensorflow import TensorflowVector
from declearn.model.torch import TorchVector
from declearn.optimizer.modules import OptiModule
from declearn.optimizer.regularizers import Regularizer


Framework = Literal["numpy", "tflow", "torch"]
FRAMEWORKS = ["numpy", "tflow", "torch"]  # type: List[Framework]
Plugin = Union[OptiModule, Regularizer]


class GradientsTestCase:
    """Framework-parametrized OptiModule testing fixtures provider."""

    def __init__(self, framework: Framework) -> None:
        """Instantiate the parametrized test-case."""
        self.framework = framework

    @property
    def vector_cls(self) -> Type[Vector]:
        """Vector subclass suitable to the tested framework."""
        classes = {
            "numpy": NumpyVector,
            "tflow": TensorflowVector,
            "torch": TorchVector,
        }  # type: Dict[str, Type[Vector]]
        return classes[self.framework]

    def convert(self, array: np.ndarray) -> ArrayLike:
        """Convert an input numpy array to a framework-based structure."""
        functions = {
            "numpy": np.array,
            "tflow": tf.convert_to_tensor,
            "torch": torch.from_numpy,  # pylint: disable=no-member
        }
        return functions[self.framework](array)  # type: ignore

    def to_numpy(self, array: ArrayLike) -> np.ndarray:
        """Convert an input framework-based structure to a numpy array."""
        if isinstance(array, np.ndarray):
            return array
        return array.numpy()  # type: ignore

    @property
    def mock_gradient(self) -> Vector:
        """Instantiate a Vector with random-valued mock gradients.

        Note: the RNG used to generate gradients has a fixed seed,
              to that gradients have the same values whatever the
              tensor framework used is.
        """
        rng = np.random.default_rng(seed=0)
        shapes = [(64, 32), (32,), (32, 16), (16,), (16, 1), (1,)]
        values = [rng.normal(size=shape) for shape in shapes]
        return self.vector_cls(
            {str(idx): self.convert(value) for idx, value in enumerate(values)}
        )


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

    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_run(self, cls: Type[Plugin], framework: Framework) -> None:
        """Test an Regularizer's run method using a given framework.

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

    def test_run_equivalence(self, cls: Type[Plugin]) -> None:
        """Test that an Regularizer's run is equivalent for all frameworks.

        Note: If a framework's run fails, warn about it but keep going,
              as `test_run` is sufficient to report framework failure.
              If no framework was correctly run, fail this test.
        """
        # Collect outputs from a newly-created plugin for each framework.
        results = []  # type: List[NumpyVector]
        for framework in FRAMEWORKS:
            f_case = GradientsTestCase(framework)
            plugin = cls()
            try:
                _, output = self._run_plugin(plugin, f_case)
            except Exception:  # pylint: disable=broad-except
                warnings.warn(
                    f"Skipping framework '{framework}' in equivalence test."
                )
            finally:
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
