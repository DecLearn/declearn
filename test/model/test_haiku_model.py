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

"""Unit tests for HaikuModel."""

import os
import warnings
from typing import Any, Callable, Dict, List, Literal, Union

import numpy as np
import pytest

try:
    import haiku as hk
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:
    pytest.skip("jax and/or haiku are unavailable", allow_module_level=True)

from declearn.model.haiku import HaikuModel, JaxNumpyVector
from declearn.test_utils import make_importable
from declearn.typing import Batch
from declearn.utils import set_device_policy

# relative imports from `model_testing.py`
with make_importable(os.path.dirname(__file__)):
    from model_testing import ModelTestCase, ModelTestSuite

# Enable float64 support.
jax.config.update("jax_enable_x64", True)


def cnn_fn(inputs: jax.Array) -> jax.Array:
    """Simple CNN in a purely functional form"""
    stack = [
        hk.Conv2D(output_channels=32, kernel_shape=(7, 7), padding="SAME"),
        jax.nn.relu,
        hk.MaxPool(window_shape=(8, 8, 1), strides=(8, 8, 1), padding="VALID"),
        hk.Conv2D(output_channels=16, kernel_shape=(5, 5), padding="SAME"),
        jax.nn.relu,
        hk.AvgPool(window_shape=(8, 8, 1), strides=(8, 8, 1), padding="VALID"),
        hk.Reshape((16,)),
        hk.Linear(1),
    ]
    model = hk.Sequential(stack)  # type: ignore
    return model(inputs)


def mlp_fn(inputs: jax.Array) -> jax.Array:
    """Simple MLP in a purely functional form"""
    model = hk.nets.MLP([32, 16, 1])
    return model(inputs)


def rnn_fn(inputs: jax.Array) -> jax.Array:
    """Simple RNN in a purely functional form"""
    inputs = inputs[None, :] if len(inputs.shape) == 1 else inputs
    core = hk.DeepRNN(
        [
            hk.Embed(100, 32),
            hk.LSTM(32),
            jax.nn.tanh,
        ]
    )
    batch_size = inputs.shape[0]
    initial_state = core.initial_state(batch_size)
    logits, _ = hk.dynamic_unroll(
        core, inputs, initial_state, time_major=False
    )
    return hk.Linear(1)(logits)[:, -1, :]


def loss_fn(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
    """Per-sample binary cross entropy"""
    y_pred = jax.nn.sigmoid(y_pred)
    y_pred = jnp.squeeze(y_pred)
    log_p, log_not_p = jnp.log(y_pred), jnp.log(1.0 - y_pred)
    return -y_true * log_p - (1.0 - y_true) * log_not_p


class HaikuTestCase(ModelTestCase):
    """Tensorflow Keras test-case-provider fixture.

    Implemented architectures are:
    * "MLP":
        - input: 64-dimensional features vectors
        - stack: 32-neurons fully-connected layer with ReLU
                 16-neurons fully-connected layer with ReLU
                 1 output neuron with sigmoid activation
    * "RNN":
        - input: 128-tokens-sequence in a 100-tokens-vocabulary
        - stack: 32-dimensional embedding matrix
                 16-neurons LSTM layer with tanh activation
                 1 output neuron with sigmoid activation
    * "CNN":
        - input: 64x64 image with 3 channels (normalized values)
        - stack: 32 7x7 conv. filters, then 8x8 max pooling
                 16 5x5 conv. filters, then 8x8 avg pooling
                 1 output neuron with sigmoid activation
    """

    vector_cls = JaxNumpyVector
    tensor_cls = jax.Array
    framework = "jax"

    def __init__(
        self,
        kind: Literal["MLP", "MLP-tune", "RNN", "CNN"],
        device: Literal["cpu", "gpu"],
    ) -> None:
        """Specify the desired model architecture."""
        if kind not in ("MLP", "MLP-tune", "RNN", "CNN"):
            raise ValueError(f"Invalid test architecture: '{kind}'.")
        if device not in ("cpu", "gpu"):
            raise ValueError(f"Invalid device choice for test: '{device}'.")
        self.kind = kind
        self.device = device
        set_device_policy(gpu=(device == "gpu"), idx=0)

    @property
    def dataset(
        self,
    ) -> List[Batch]:
        """Suited toy binary-classification dataset."""
        # Generate data using numpy.
        rng = np.random.default_rng(seed=0)
        if self.kind.startswith("MLP"):
            inputs = rng.normal(size=(2, 32, 64)).astype("float32")
        elif self.kind == "RNN":
            inputs = rng.choice(100, size=(2, 32, 128))
        elif self.kind == "CNN":
            inputs = rng.normal(size=(2, 32, 64, 64, 3)).astype("float32")
        else:
            raise ValueError("Invalid model 'kind'.")
        labels = rng.choice(2, size=(2, 32))
        # Convert that data to jax-numpy and return it.
        with warnings.catch_warnings():  # jax.jit(device=...) is deprecated
            warnings.simplefilter("ignore", DeprecationWarning)
            convert = jax.jit(jnp.asarray, backend=self.device)
            batches = list(
                zip(
                    convert(inputs),  # pylint: disable=not-callable
                    convert(labels),  # pylint: disable=not-callable
                    [None, None],
                    strict=False,
                )
            )
        return batches  # type: ignore

    @property
    def model(self) -> HaikuModel:
        """Suited toy binary-classification haiku models."""
        if self.kind == "CNN":
            shape = [64, 64, 3]
            model_fn = cnn_fn
        elif self.kind.startswith("MLP"):
            shape = [64]
            model_fn = mlp_fn
        elif self.kind == "RNN":
            shape = [128]
            model_fn = rnn_fn
        else:
            raise ValueError("Invalid model 'kind'.")
        model = HaikuModel(model_fn, loss_fn)
        model.initialize(
            {
                "features_shape": shape,
                "data_type": "int" if self.kind == "RNN" else "float32",
            }
        )
        if self.kind == "MLP-tune":
            names = model.get_weight_names()
            model.set_trainable_weights([names[i] for i in range(3)])
        return model

    def assert_correct_device(
        self,
        vector: JaxNumpyVector,
    ) -> None:
        """Raise if a vector is backed on the wrong type of device."""
        name = f"{self.device}:0"
        assert all(
            f"{device.platform}:{device.id}" == name
            for arr in vector.coefs.values()
            for device in arr.devices()
        )

    def get_trainable_criterion(
        self,
        c_type: str,
    ) -> Union[
        List[str],
        Dict[str, Dict[str, Any]],
        Callable[[str, str, jax.Array], bool],
    ]:
        "Build different weight freezing criteria"
        if c_type == "names":
            names = self.model.get_weight_names()
            return [names[2], names[3]]
        if c_type == "pytree":
            params = self.model._params
            return {k: v for i, (k, v) in enumerate(params.items()) if i != 1}
        if c_type == "predicate":
            return lambda m, n, p: n != "b"
        raise KeyError(f"Invalid 'c_type' parameter: {c_type}.")


@pytest.fixture(name="test_case")
def fixture_test_case(
    kind: Literal["MLP", "MLP-tune", "RNN", "CNN"],
    device: Literal["cpu", "gpu"],
    cpu_only: bool,
) -> HaikuTestCase:
    """Fixture to access a TensorflowTestCase."""
    if cpu_only and (device == "gpu"):
        pytest.skip(reason="--cpu-only mode")
    return HaikuTestCase(kind, device)


DEVICES = ["cpu"]
try:
    jax.devices("gpu")
except RuntimeError:
    pass
else:
    DEVICES.append("gpu")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["MLP", "MLP-tune", "RNN", "CNN"])
class TestHaikuModel(ModelTestSuite):
    """Unit tests for declearn.model.tensorflow.TensorflowModel."""

    @pytest.mark.filterwarnings("ignore: Our custom Haiku serialization")
    def test_get_config(
        self,
        test_case: ModelTestCase,
    ) -> None:
        super().test_get_config(test_case)

    @pytest.mark.filterwarnings("ignore: Our custom Haiku serialization")
    def test_from_config(
        self,
        test_case: ModelTestCase,
    ) -> None:
        super().test_from_config(test_case)

    @pytest.mark.parametrize(
        "criterion_type", ["names", "pytree", "predicate"]
    )
    def test_get_frozen_weights(
        self,
        test_case: HaikuTestCase,
        criterion_type: str,
    ) -> None:
        """Check that `get_weights` behaves properly with frozen weights."""
        model: HaikuModel = test_case.model
        criterion = test_case.get_trainable_criterion(criterion_type)
        model.set_trainable_weights(criterion)  # freeze some weights
        w_all = model.get_weights()
        w_trn = model.get_weights(trainable=True)
        assert set(w_trn.coefs).issubset(w_all.coefs)  # check on keys
        n_params = len(model.get_weight_names())
        n_trainable = len(model.get_weight_names(trainable=True))
        assert n_trainable < n_params
        assert len(w_trn.coefs) == n_trainable
        assert len(w_all.coefs) == n_params

    @pytest.mark.parametrize(
        "criterion_type", ["names", "pytree", "predicate"]
    )
    def test_set_frozen_weights(
        self,
        test_case: HaikuTestCase,
        criterion_type: str,
    ) -> None:
        """Check that `set_weights` behaves properly with frozen weights."""
        # similar code to TorchModel tests; pylint: disable=duplicate-code
        # Setup a model with some frozen weights, and gather trainable ones.
        model = test_case.model
        criterion = test_case.get_trainable_criterion(criterion_type)
        model.set_trainable_weights(criterion)  # freeze some weights
        w_trn = model.get_weights(trainable=True)
        # Test that `set_weights` works if and only if properly parametrized.
        with pytest.raises(KeyError):
            model.set_weights(w_trn)
        with pytest.raises(KeyError):
            model.set_weights(model.get_weights(), trainable=True)
        model.set_weights(w_trn, trainable=True)

    def test_proper_model_placement(
        self,
        test_case: HaikuTestCase,
    ) -> None:
        """Check that at instantiation, model weights are properly placed."""
        model = test_case.model
        policy = model.device_policy
        assert policy.gpu == (test_case.device == "gpu")
        assert policy.idx == 0
        params = jax.tree_util.tree_leaves(model._params)
        device = f"{test_case.device}:0"
        for arr in params:
            assert len(arr.devices()) == 1
            arr_dev = list(arr.devices())[0]
            assert f"{arr_dev.platform}:{arr_dev.id}" == device
