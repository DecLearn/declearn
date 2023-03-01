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

"""Unit tests for the TensorflowOptiModule class."""

import sys
import warnings
from typing import Iterator, Type, Union

import numpy as np
import pytest

# pylint: disable=duplicate-code
try:
    with warnings.catch_warnings():  # silence tensorflow import-time warnings
        warnings.simplefilter("ignore")
        import tensorflow as tf  # type: ignore
except ModuleNotFoundError:
    pytest.skip("TensorFlow is unavailable", allow_module_level=True)
# pylint: enable=duplicate-code

from declearn.model.tensorflow import TensorflowOptiModule, TensorflowVector
from declearn.optimizer.modules import OptiModule
from declearn.test_utils import GradientsTestCase
from declearn.utils import set_device_policy


# dirty trick to import from `model_testing.py`;
# fmt: off
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(".")
from test_modules import OptiModuleTestSuite
sys.path.pop()
# fmt: on


set_device_policy(gpu=False)  # force most tests to run on CPU


DEVICES = ["CPU"]
if tf.config.list_logical_devices("GPU"):
    DEVICES.append("GPU")


@pytest.fixture(name="cls")
def cls_fixture(optim: str) -> Iterator[Type[TensorflowOptiModule]]:
    """Fixture to provide with preset TensorflowOptiModule constructors."""
    defaults = TensorflowOptiModule.__init__.__defaults__
    TensorflowOptiModule.__init__.__defaults__ = (optim,)
    yield TensorflowOptiModule
    TensorflowOptiModule.__init__.__defaults__ = defaults


def get_optim_dec(optim: str) -> OptiModule:
    """Instanciate an OptiModule parameterized to match a torch one."""
    if optim == "adam":
        name = "adam"
        kwargs = {"beta_1": 0.9, "beta_2": 0.999, "eps": 1e-07}
    elif optim == "rmsprop":
        name = "rmsprop"
        kwargs = {"beta": 0.9, "eps": 0.0}
        # We are disabling the use of epsilon, because keras rmsprop divices
        # outputs by sqrt(state + epsilon) instead of (sqrt(state) + epsilon)
        # which can only be corrected by pre-computing state variables, which
        # is tedious and greatly diminishes the interest for the test.
    elif optim == "adagrad":
        name = "adagrad"
        # Adjust the initial state and epsilon for a couple of reasons:
        # 1. `initial_accumulator_value=0.1` in tensorflow, but 0. in declearn
        # 2. Keras Adagrad divides outputs by sqrt(state + epsilon) instead of
        #    (sqrt(state) + epsilon). Hence here we force declearn to follow
        #    the keras formula, which is simpler than the other way around.
        module = OptiModule.from_specs(name, {"eps": 0.0})
        module.state = 0.1 + 1e-07  # type: ignore
        return module
    else:
        raise KeyError(f"Invalid 'optim' fixture parameter value: '{optim}'.")
    return OptiModule.from_specs(name, kwargs)


def fix_adam_epsilon(
    module: TensorflowOptiModule,
) -> TensorflowOptiModule:
    """Fix the epsilon parameter of a wrapped keras Adam optimizer.

    In Keras, Adam does not use epsilon as it should: the epsilon parameter
    is treated as though it were epsilon-hat, which means that to implement
    the canonical formula one must update epsilon manually at each step, as
    at step t `epsilon-hat := epsilon * sqrt(1 - beta_2^t)`.

    See this issue: https://github.com/keras-team/keras/issues/17391

    This function therefore updates the epsilon parameter of the keras adam
    optimizer wrapped into a TensorflowOptiModule. It assumes that epsilon
    was already adjusted at all prior steps.
    """
    idx = module.optim.iterations.numpy()
    prev = np.sqrt(1 - module.optim.beta_2**idx) if idx else 1.0
    curr = np.sqrt(1 - module.optim.beta_2 ** (idx + 1))
    module.optim.epsilon = module.optim.epsilon * curr / prev
    return module


def to_numpy(tensor: Union[tf.Tensor, tf.IndexedSlices]) -> np.ndarray:
    """Convert a tensorflow Tensor or IndexedSlices to numpy."""
    if isinstance(tensor, tf.IndexedSlices):
        return tensor.values.numpy()
    return tensor.numpy()


@pytest.fixture(name="framework")
def framework_fixture():
    """Fixture to ensure 'TensorflowOptiModule' only receives tf gradients."""
    return "tensorflow"


@pytest.mark.parametrize("optim", ["adam", "rmsprop", "adagrad"])
class TestTensorflowOptiModule(OptiModuleTestSuite):
    """Unit tests for declearn.model.torch.TensorflowOptiModule."""

    def test_run_equivalence(  # type: ignore
        self, cls: Type[OptiModule]
    ) -> None:
        # This test is undefined for a framework-specific plugin.
        pass

    def test_equivalent_declearn(self, optim: str) -> None:
        """Test that declearn modules are equivalent to keras ones.

        Instantiate a TensorflowOptiModule wrapping a keras Optimizer,
        as well as a declearn OptiModule that matches its configuration
        (adjusting hyper-parameters to tensorflow's defaults).

        Ensure that on 10 successive passes on the same random-valued
        input gradients, outputs have the same values, up to numerical
        precision (relative tolerance of 10^-5, absolute of 10^-8).
        """
        optim_tfk = TensorflowOptiModule(optim)
        optim_dec = get_optim_dec(optim)
        gradients = GradientsTestCase("tensorflow").mock_gradient
        if optim == "rmsprop":  # disable epsilon due to keras formula error
            optim_tfk.optim.epsilon = 0.0
        # Compare the declearn and keras implementations over 10 steps.
        for _ in range(10):
            # Run Adam-specific fix.
            if optim == "adam":
                optim_tfk = fix_adam_epsilon(optim_tfk)
            # Compute gradients with both implementations.
            grads_tfk = optim_tfk.run(gradients).coefs
            grads_dec = optim_dec.run(gradients).coefs
            # Assert that the output gradients are (nearly) identical.
            assert all(
                np.allclose(to_numpy(grads_tfk[key]), to_numpy(grads_dec[key]))
                for key in gradients.coefs
            )

    def test_reset(self, optim: str) -> None:
        """Test that the `TensorflowOptiModule.reset` method works."""
        # Set up a module and two sets of differently-labeled gradients.
        module = TensorflowOptiModule(optim)
        grads_a = GradientsTestCase("tensorflow").mock_gradient
        grads_b = TensorflowVector(
            {f"{key}_bis": val for key, val in grads_a.coefs.items()}
        )
        # Check that running inconsistent gradients fails.
        outpt_a = module.run(grads_a)
        assert isinstance(outpt_a, TensorflowVector)
        with pytest.raises(KeyError):
            module.run(grads_b)
        # Test that resetting enables setting up a new input spec.
        module.reset()
        outpt_b = module.run(grads_b)
        assert isinstance(outpt_b, TensorflowVector)
        # Check that results are indeed the same, save for their names.
        # This means inner states have been properly reset.
        outpt_a = TensorflowVector(
            {f"{key}_bis": val for key, val in outpt_a.coefs.items()}
        )
        assert outpt_b == outpt_a

    @pytest.mark.parametrize("device", DEVICES)
    def test_device_placement(self, optim: str, device: str) -> None:
        """Test that the optimizer and computations are properly placed."""
        # Set the device policy, setup a module and run computations.
        set_device_policy(gpu=(device == "GPU"), idx=None)
        module = TensorflowOptiModule(optim)
        with tf.device(device):
            grads = GradientsTestCase("tensorflow").mock_gradient
        updts = module.run(grads)
        # Assert that the outputs and internal states are properly placed.
        assert all(device in t.device for t in updts.coefs.values())
        assert all(device in t.device for t in module.optim.variables())
        # Reset device policy to run other tests on CPU as expected.
        set_device_policy(gpu=False)
