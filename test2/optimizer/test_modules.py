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
import tempfile
import warnings
from typing import Dict, List, Type

import pytest
import numpy as np
with warnings.catch_warnings():  # silence tensorflow import-time warnings
    warnings.simplefilter("ignore")
    import tensorflow as tf  # type: ignore
import torch
from numpy.typing import ArrayLike
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn2.model.api import NumpyVector, Vector
from declearn2.model.tensorflow import TensorflowVector
from declearn2.model.torch import TorchVector
from declearn2.optimizer.modules import OptiModule
from declearn2.utils._register import REGISTRIES


# unproper but efficient way to list modules; pylint: disable=protected-access
OPTIMODULE_SUBCLASSES = REGISTRIES["OptiModule"]._reg
# pylint: enable=protected-access

Framework = Literal["numpy", "tflow", "torch"]
FRAMEWORKS = ["numpy", "tflow", "torch"]  # type: List[Framework]


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
        return self.vector_cls({
            str(idx): self.convert(value) for idx, value in enumerate(values)
        })


@pytest.mark.parametrize(
    'cls', OPTIMODULE_SUBCLASSES.values(), ids=OPTIMODULE_SUBCLASSES.keys()
)
class TestOptiModule:
    """Unit tests for declearn.optimizer.modules.OptiModule subclasses."""

    @staticmethod
    def assert_equivalent(mod_a: OptiModule, mod_b: OptiModule) -> None:
        """Assert that two modules are of same type and configuration."""
        assert type(mod_a) is type(mod_b)
        assert mod_a.get_config() == mod_b.get_config()

    def test_config(self, cls: Type[OptiModule]) -> None:
        """Test an OptiModule's (get|from)_config methods."""
        module = cls()
        cfg = module.get_config()
        self.assert_equivalent(module, cls.from_config(cfg))

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

    @pytest.mark.parametrize('framework', FRAMEWORKS)
    def test_run(self, cls: Type[OptiModule], framework: Framework) -> None:
        """Test an OptiModule's run method using a given framework.

        Note: Only check that input and output gradients have
              same specs, not their algorithmic correctness.
        """
        test_case = GradientsTestCase(framework)
        module = cls()
        inputs = test_case.mock_gradient
        output = module.run(inputs)
        assert isinstance(output, test_case.vector_cls)
        assert output.coefs.keys() == inputs.coefs.keys()
        # In fact, check compatibility between inputs and outputs.
        assert isinstance(inputs + output, test_case.vector_cls)

    def test_run_equivalence(self, cls: Type[OptiModule]) -> None:
        """Test that an OptiModule's run is equivalent for all frameworks.

        Note: If a framework's run fails, warn about it but keep going,
              as `test_run` is sufficient to report framework failure.
              If no framework was correctly run, fail this test.
        """
        # Collect outputs from a newly-created module for each framework.
        results = []  # type: List[NumpyVector]
        for framework in FRAMEWORKS:
            f_case = GradientsTestCase(framework)
            module = cls()
            inputs = f_case.mock_gradient
            try:
                output = module.run(inputs)
            except Exception:  # pylint: disable=broad-except
                warnings.warn(
                    f"Skipping framework '{framework}' in equivalence test."
                )
            finally:
                results.append(NumpyVector({
                    key: f_case.to_numpy(val)
                    for key, val in output.coefs.items()
                }))
        # Check that all collected results have the same values.
        assert results
        assert all(
            np.allclose(results[0].coefs[key], val)  # for numerical precision
            for res in results[1:]
            for key, val in res.coefs.items()
        )

    def test_collect_aux_var(self, cls: Type[OptiModule]) -> None:
        """Test an OptiModule's collect_aux_var method."""
        module = cls()
        aux_var = module.collect_aux_var()
        assert (aux_var is None) or isinstance(aux_var, dict)
        if isinstance(aux_var, dict):
            assert isinstance(json.dumps(aux_var), str)
