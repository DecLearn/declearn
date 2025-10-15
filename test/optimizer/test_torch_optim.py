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

"""Unit tests for the TorchOptiModule class."""

import importlib
import os
from typing import Iterator, Type
from unittest import mock

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("Torch is unavailable", allow_module_level=True)

from declearn.model.torch import TorchOptiModule, TorchVector
from declearn.optimizer.modules import OptiModule
from declearn.test_utils import GradientsTestCase, make_importable
from declearn.utils import set_device_policy

# relative imports from `test_modules.py`
with make_importable(os.path.dirname(__file__)):
    from test_modules import OptiModuleTestSuite


DEVICES = ["CPU"]
if torch.cuda.device_count():
    DEVICES.append("GPU")


@pytest.fixture(name="optim_cls")
def optim_cls_fixture(optim: str) -> str:
    """Fixture to provide with a torch Optimizer's import path."""
    return {
        "adam": "torch.optim.Adam",
        "rmsprop": "torch.optim.RMSprop",
        "adagrad": "torch.optim.Adagrad",
    }[optim]


@pytest.fixture(name="cls")
def cls_fixture(optim_cls: str) -> Iterator[Type[TorchOptiModule]]:
    """Fixture to provide with preset TorchOptiModule constructors."""
    defaults = TorchOptiModule.__init__.__defaults__
    TorchOptiModule.__init__.__defaults__ = (optim_cls, False)
    with mock.patch("builtins.input") as patch_input:
        patch_input.return_value = "y"
        yield TorchOptiModule
    TorchOptiModule.__init__.__defaults__ = defaults


def get_optim_dec(optim_cls: str) -> OptiModule:
    """Instanciate an OptiModule parameterized to match a torch one."""
    optim = optim_cls.rsplit(".", 1)[-1].lower()
    if optim == "adam":
        name = "adam"
        kwargs = {"beta_1": 0.9, "beta_2": 0.999, "eps": 1e-08}
    elif optim == "rmsprop":
        name = "rmsprop"
        kwargs = {"beta": 0.99, "eps": 1e-08}
    elif optim == "adagrad":
        name = "adagrad"
        kwargs = {"eps": 1e-10}
    else:
        raise KeyError(f"Invalid 'optim' fixture parameter value: '{optim}'.")
    return OptiModule.from_specs(name, kwargs)


@pytest.fixture(name="framework")
def framework_fixture():
    """Fixture to ensure 'TorchOptiModule' only receives torch gradients."""
    return "torch"


@pytest.mark.parametrize("optim", ["adam", "rmsprop", "adagrad"])
class TestTorchOptiModule(OptiModuleTestSuite):
    """Unit tests for declearn.model.torch.TorchOptiModule."""

    def test_run_equivalence(self, cls: Type[OptiModule]) -> None:
        # This test is undefined for a framework-specific plugin.
        pass

    def test_validate_torch(self, optim_cls: str) -> None:
        """Test that user-validation of torch modules' import is skipped."""
        with mock.patch("builtins.input") as patch_input:
            patch_input.return_value = "y"
            TorchOptiModule(optim_cls, validate=True)
            patch_input.assert_not_called()

    def test_equivalent_declearn(self, optim_cls: str) -> None:
        """Test that declearn modules are equivalent to torch ones.

        Instantiate a TorchOptiModule wrapping a torch.optim.Optimizer,
        as well as a declearn OptiModule that matches its configuration
        (adjusting hyper-parameters to torch's defaults).

        Ensure that on 10 successive passes on the same random-valued
        input gradients, outputs have the same values, up to numerical
        precision (relative tolerance of 10^-5, absolute of 10^-8).
        """
        optim_pyt = TorchOptiModule(optim_cls, validate=False)
        optim_dec = get_optim_dec(optim_cls)
        gradients = GradientsTestCase("torch").mock_gradient
        for _ in range(10):
            grads_pyt = optim_pyt.run(gradients).coefs
            grads_dec = optim_dec.run(gradients).coefs
            assert all(
                np.allclose(grads_pyt[key].numpy(), grads_dec[key].numpy())
                for key in gradients.coefs
            )

    def test_reset(self, optim_cls: str) -> None:
        """Test that the `TorchOptiModule.reset` method works."""
        # Set up a module and two sets of differently-labeled gradients.
        module = TorchOptiModule(optim_cls, validate=False)
        grads_a = GradientsTestCase("torch").mock_gradient
        grads_b = TorchVector(
            {f"{key}_bis": val for key, val in grads_a.coefs.items()}
        )
        # Check that running inconsistent gradients fails.
        outpt_a = module.run(grads_a)
        assert isinstance(outpt_a, TorchVector)
        with pytest.raises(KeyError):
            module.run(grads_b)
        # Test that resetting enables setting up a new input spec.
        module.reset()
        outpt_b = module.run(grads_b)
        assert isinstance(outpt_b, TorchVector)
        # Check that results are indeed the same, save for their names.
        # This means inner states have been properly reset.
        outpt_a = TorchVector(
            {f"{key}_bis": val for key, val in outpt_a.coefs.items()}
        )
        assert outpt_b == outpt_a

    # similarity with tensorflow counterpart; pylint: disable=duplicate-code

    @pytest.mark.parametrize("device", DEVICES)
    def test_device_placement(
        self,
        optim_cls: str,
        device: str,
        cpu_only: bool,
    ) -> None:
        """Test that the optimizer and computations are properly placed."""
        if cpu_only and device == "GPU":
            pytest.skip(reason="--cpu-only mode")
        # Set the device policy, setup a module and run computations.
        set_device_policy(gpu=(device == "GPU"), idx=None)
        module = TorchOptiModule(optim_cls)
        grads = GradientsTestCase("torch").mock_gradient
        if device == "GPU":
            for key, tensor in grads.coefs.items():
                grads.coefs[key] = tensor.cuda()
        updts = module.run(grads)
        # Assert that the outputs and internal states are properly placed.
        dtype = "cuda" if device == "GPU" else "cpu"
        assert all(t.device.type == dtype for t in updts.coefs.values())
        state = module.get_state()["state"]
        assert all(
            tensor.device.type == dtype
            for _, group in state
            for tensor in group
            if isinstance(tensor, torch.Tensor)
        )
        # Reset device policy to run other tests on CPU as expected.
        set_device_policy(gpu=False)

    # pylint: enable=duplicate-code


class FakeOptimizer(torch.optim.Optimizer):
    """Fake torch Optimizer subclass."""

    step = mock.create_autospec(torch.optim.Optimizer.step)


class EmptyClass:
    """Empty class to test non-torch-optimizer imports' failure."""

    # mock class; pylint: disable=too-few-public-methods

    __init__ = mock.MagicMock()


class TestTorchOptiModuleValidateImports:
    """Test that user-validation of third-party modules' import works."""

    def test_validate_accept(self) -> None:
        """Assert that (fake) user inputs enable validating imports."""
        # Set up the import string for FakeOptimizer and pre-import its module.
        string = f"{__name__}.FakeOptimizer"
        module = importlib.import_module(__name__)
        # Run the TorchOptiModule instantiation with patched objects.
        with mock.patch("builtins.input") as patch_input:
            with mock.patch("importlib.import_module") as patch_import:
                patch_input.return_value = "y"
                patch_import.return_value = module
                optim = TorchOptiModule(string, validate=True)
        # Assert the expected calls were made and FakeOptimizer was assigned.
        patch_input.assert_called_once()
        patch_import.assert_called_once_with(__name__)
        assert optim.optim_cls is FakeOptimizer

    def test_validate_deny(self) -> None:
        """Assert (fake) user inputs enable blocking imports."""
        # Run the TorchOptiModule instantiation with fake user denial command.
        with mock.patch("builtins.input") as patch_input:
            with mock.patch("importlib.import_module") as patch_import:
                patch_input.return_value = "n"
                with pytest.raises(RuntimeError):
                    TorchOptiModule(f"{__name__}.FakeOptimizer", validate=True)
        # Assert the expected calls were made (or not).
        patch_input.assert_called_once()
        patch_import.assert_not_called()

    def test_validate_wrong_class(self) -> None:
        """Assert importing an invalid class raises a TypeError."""
        # Set up the import string for EmptyClass and pre-import its module.
        string = f"{__name__}.EmptyClass"
        module = importlib.import_module(__name__)
        # Run the TorchOptiModule instantiation with patched objects.
        with mock.patch("builtins.input") as patch_input:
            with mock.patch("importlib.import_module") as patch_import:
                patch_input.return_value = "y"
                patch_import.return_value = module
                with pytest.raises(TypeError):
                    TorchOptiModule(string, validate=True)
        # Assert the expected calls were made.
        patch_input.assert_called_once()
        patch_import.assert_called_once_with(__name__)
