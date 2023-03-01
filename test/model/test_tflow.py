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

"""Unit tests for TensorflowModel."""

import warnings
import sys
from typing import Any, List, Literal

import numpy as np
import pytest

try:
    with warnings.catch_warnings():  # silence tensorflow import-time warnings
        warnings.simplefilter("ignore")
        import tensorflow as tf  # type: ignore
except ModuleNotFoundError:
    pytest.skip("TensorFlow is unavailable", allow_module_level=True)

from declearn.model.tensorflow import TensorflowModel, TensorflowVector
from declearn.typing import Batch
from declearn.utils import set_device_policy

# dirty trick to import from `model_testing.py`;
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(".")
from model_testing import ModelTestSuite, ModelTestCase


class TensorflowTestCase(ModelTestCase):
    """Tensorflow Keras test-case-provider fixture.

    Implemented architectures are:
    * "MLP":
        - input: 64-dimensional features vectors
        - stack: 32-neurons fully-connected layer with ReLU
                 16-neurons fully-connected layer with ReLU
                 1 output neuron with sigmoid activation
    * "MLP-tune":
        - same as NLP, but freeze the first layer of the stack
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

    vector_cls = TensorflowVector
    tensor_cls = tf.Tensor

    def __init__(
        self,
        kind: Literal["MLP", "MLP-tune", "RNN", "CNN"],
        device: Literal["CPU", "GPU"],
    ) -> None:
        """Specify the desired model architecture."""
        if kind not in ("MLP", "MLP-tune", "RNN", "CNN"):
            raise ValueError(f"Invalid keras test architecture: '{kind}'.")
        if device not in ("CPU", "GPU"):
            raise ValueError(f"Invalid device choice for test: '{device}'.")
        self.kind = kind
        self.device = device
        set_device_policy(gpu=(device == "GPU"), idx=0)

    @staticmethod
    def to_numpy(
        tensor: Any,
    ) -> np.ndarray:
        """Convert an input tensor to a numpy array."""
        if isinstance(tensor, tf.IndexedSlices):
            tensor = tf.convert_to_tensor(tensor)
        assert isinstance(tensor, tf.Tensor)
        return tensor.numpy()

    @property
    def dataset(
        self,
    ) -> List[Batch]:
        """Suited toy binary-classification dataset."""
        rng = tf.random.get_global_generator()
        rng.reset_from_seed(0)
        if self.kind.startswith("MLP"):
            inputs = rng.normal((2, 32, 64))
        elif self.kind == "RNN":
            inputs = rng.uniform((2, 32, 128), 0, 100, tf.int32)
        elif self.kind == "CNN":
            inputs = rng.normal((2, 32, 64, 64, 3))
        labels = rng.uniform((2, 32), 0, 2, tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, None))
        return list(iter(dataset))

    @property
    def model(
        self,
    ) -> TensorflowModel:
        """Suited toy binary-classification keras model."""
        if self.kind.startswith("MLP"):
            stack = [
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
            shape = [None, 64]
            if self.kind == "MLP-tune":
                stack[0].trainable = False
        elif self.kind == "RNN":
            stack = [
                tf.keras.layers.Embedding(100, 32),
                tf.keras.layers.LSTM(16, activation="tanh"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
            shape = [None, 128]
        elif self.kind == "CNN":
            cnn_kwargs = {"padding": "same", "activation": "relu"}
            stack = [
                tf.keras.layers.Conv2D(32, 7, **cnn_kwargs),
                tf.keras.layers.MaxPool2D((8, 8)),
                tf.keras.layers.Conv2D(16, 5, **cnn_kwargs),
                tf.keras.layers.AveragePooling2D((8, 8)),
                tf.keras.layers.Reshape((16,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
            shape = [None, 64, 64, 3]
        tfmod = tf.keras.Sequential(stack)
        tfmod.build(shape)  # as model is built, no data_info is required
        return TensorflowModel(tfmod, loss="binary_crossentropy", metrics=None)

    def assert_correct_device(
        self,
        vector: TensorflowVector,
    ) -> None:
        """Raise if a vector is backed on the wrong type of device."""
        name = f"{self.device}:0"
        assert all(
            tensor.device.endswith(name) for tensor in vector.coefs.values()
        )


@pytest.fixture(name="test_case")
def fixture_test_case(
    kind: Literal["MLP", "MLP-tune", "RNN", "CNN"],
    device: Literal["CPU", "GPU"],
) -> TensorflowTestCase:
    """Fixture to access a TensorflowTestCase."""
    return TensorflowTestCase(kind, device)


DEVICES = ["CPU"]
if tf.config.list_logical_devices("GPU"):
    DEVICES.append("GPU")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["MLP", "MLP-tune", "RNN", "CNN"])
class TestTensorflowModel(ModelTestSuite):
    """Unit tests for declearn.model.tensorflow.TensorflowModel."""

    def test_get_frozen_weights(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that `get_weights` behaves properly with frozen weights."""
        model = test_case.model
        tfmod = getattr(model, "_model")  # type: tf.keras.Sequential
        tfmod.layers[0].trainable = False  # freeze the first layer's weights
        w_all = model.get_weights()
        w_trn = model.get_weights(trainable=True)
        assert set(w_trn.coefs).issubset(w_all.coefs)  # check on keys
        assert w_trn.coefs.keys() == {v.name for v in tfmod.trainable_weights}
        assert w_all.coefs.keys() == {v.name for v in tfmod.weights}

    def test_set_frozen_weights(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that `set_weights` behaves properly with frozen weights."""
        # Setup a model with some frozen weights, and gather trainable ones.
        model = test_case.model
        tfmod = getattr(model, "_model")  # type: tf.keras.Sequential
        tfmod.layers[0].trainable = False  # freeze the first layer's weights
        w_trn = model.get_weights(trainable=True)
        # Test that `set_weights` works if and only if properly parametrized.
        with pytest.raises(KeyError):
            model.set_weights(w_trn)
        model.set_weights(w_trn, trainable=True)
        with pytest.raises(KeyError):
            model.set_weights(model.get_weights(), trainable=True)

    def test_proper_model_placement(
        self,
        test_case: TensorflowTestCase,
    ) -> None:
        """Check that at instantiation, model weights are properly placed."""
        model = test_case.model
        policy = model.device_policy
        assert policy.gpu == (test_case.device == "GPU")
        assert policy.idx == 0
        tfmod = getattr(model, "_model")
        device = f"{test_case.device}:0"
        for var in tfmod.weights:
            assert var.device.endswith(device)
