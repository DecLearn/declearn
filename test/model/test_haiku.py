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

"""Unit tests for HaikuModel."""

import sys
from typing import Any, List, Literal

import numpy as np
import pytest

try:
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from jax.config import config as jaxconfig
except ModuleNotFoundError:
    pytest.skip("jax and/or haiku are unavailable", allow_module_level=True)

from declearn.model.haiku import HaikuModel, JaxNumpyVector
from declearn.typing import Batch
from declearn.utils import set_device_policy

# dirty trick to import from `model_testing.py`;
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(".")
from model_testing import ModelTestCase, ModelTestSuite

# Overriding float32 default in jax
jaxconfig.update("jax_enable_x64", True)


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
    * "CNN":
        - input: 64x64 image with 3 channels (normalized values)
        - stack: 32 7x7 conv. filters, then 8x8 max pooling
                 16 5x5 conv. filters, then 8x8 avg pooling
                 1 output neuron with sigmoid activation
    """

    vector_cls = JaxNumpyVector
    tensor_cls = jax.Array

    def __init__(
        self,
        kind: Literal["MLP", "CNN", "MLP-tune"],
        device: Literal["cpu", "gpu"],
    ) -> None:
        """Specify the desired model architecture."""
        if kind not in ("MLP", "CNN", "MLP-tune"):
            raise ValueError(f"Invalid test architecture: '{kind}'.")
        if device not in ("cpu", "gpu"):
            raise ValueError(f"Invalid device choice for test: '{device}'.")
        self.kind = kind
        self.device = device
        set_device_policy(gpu=(device == "gpu"), idx=0)

    @staticmethod
    def to_numpy(
        tensor: Any,
    ) -> np.ndarray:
        """Convert an input jax jax.Array to a numpy jax.Array."""
        assert isinstance(tensor, jax.Array)
        return np.asarray(tensor)

    @property
    def dataset(
        self,
    ) -> List[Batch]:
        """Suited toy binary-classification dataset."""
        rng = np.random.default_rng(seed=0)
        if self.kind.startswith("MLP"):
            inputs = rng.normal(size=(2, 32, 64)).astype("float32")
        elif self.kind == "CNN":
            inputs = rng.normal(size=(2, 32, 64, 64, 3)).astype("float32")
        labels = rng.choice(2, size=(2, 32))
        inputs = jnp.asarray(inputs)  # type: ignore
        labels = jnp.asarray(labels)  # type: ignore
        batches = list(zip(inputs, labels, [None, None]))
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
        model = HaikuModel(model_fn, loss_fn)
        model.initialize({"features_shape": shape, "data_type": "float32"})
        if self.kind == "MLP-tune":
            model.set_trainable_weights([0, 1, 2])
        return model

    def assert_correct_device(
        self,
        vector: JaxNumpyVector,
    ) -> None:
        """Raise if a vector is backed on the wrong type of device."""
        name = f"{self.device}:0"
        assert all(
            f"{arr.device().platform}:{arr.device().id}" == name
            for arr in vector.coefs.values()
        )

    def get_trainable_criterion(self, c_type: str):
        "Build different weight freezing criteria"
        if c_type == "indexes":
            crit = [2, 3]
        if c_type == "pytree":
            params = self.model.get_named_weights()
            params.pop(list(params.keys())[1])
            crit = params
        if c_type == "predicate":
            crit = lambda m, n, p: n != "b"  # pylint: disable=C3001
        return crit


@pytest.fixture(name="test_case")
def fixture_test_case(
    kind: Literal["MLP", "CNN", "MLP-tune"],
    device: Literal["cpu", "gpu"],
) -> HaikuTestCase:
    """Fixture to access a TensorflowTestCase."""
    return HaikuTestCase(kind, device)


DEVICES = ["cpu"]
if any(d.platform == "gpu" for d in jax.devices()):
    DEVICES.append("gpu")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["MLP", "CNN", "MLP-tune"])
class TestHaikuModel(ModelTestSuite):
    """Unit tests for declearn.model.tensorflow.TensorflowModel."""

    @pytest.mark.filterwarnings("ignore: Our custom Haiku serialization")
    def test_serialization(
        self,
        test_case: ModelTestCase,
    ) -> None:
        super().test_serialization(test_case)

    @pytest.mark.parametrize(
        "criterion_type", ["indexes", "pytree", "predicate"]
    )
    def test_get_frozen_weights(
        self,
        test_case: HaikuTestCase,
        criterion_type: str,
    ) -> None:
        """Check that `get_weights` behaves properly with frozen weights."""
        model = test_case.model  # type: HaikuModel
        criterion = test_case.get_trainable_criterion(criterion_type)
        model.set_trainable_weights(criterion)  # freeze some weights
        w_all = model.get_weights()
        w_trn = model.get_weights(trainable=True)
        assert set(w_trn.coefs).issubset(w_all.coefs)  # check on keys
        n_params = len(model._pleaves)  # pylint: disable=protected-access
        n_trainable = len(model._trainable)  # pylint: disable=protected-access
        assert n_trainable < n_params
        assert len(w_trn.coefs) == n_trainable
        assert len(w_all.coefs) == n_params

    @pytest.mark.parametrize(
        "criterion_type", ["indexes", "pytree", "predicate"]
    )
    def test_set_frozen_weights(
        self,
        test_case: HaikuTestCase,
        criterion_type: str,
    ) -> None:
        """Check that `set_weights` behaves properly with frozen weights."""
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
        params = getattr(model, "_pleaves")
        device = f"{test_case.device}:0"
        for arr in params:
            assert f"{arr.device().platform}:{arr.device().id}" == device
