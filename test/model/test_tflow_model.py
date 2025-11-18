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

"""Unit tests for TensorflowModel."""

import os
import warnings
from typing import List, Literal

import pytest

try:
    with warnings.catch_warnings():  # silence tensorflow import-time warnings
        warnings.simplefilter("ignore")
        import tensorflow as tf  # type: ignore
except ModuleNotFoundError:
    pytest.skip("TensorFlow is unavailable", allow_module_level=True)
else:
    import tensorflow.keras as tf_keras  # type: ignore

from declearn.model.tensorflow import TensorflowModel, TensorflowVector
from declearn.model.tensorflow.utils import build_keras_loss
from declearn.test_utils import make_importable
from declearn.typing import Batch
from declearn.utils import set_device_policy

# relative imports from `model_testing.py`
with make_importable(os.path.dirname(__file__)):
    from model_testing import ModelTestCase, ModelTestSuite

# mypy: ignore-errors


class TensorflowTestCase(ModelTestCase):
    """Tensorflow Keras test-case-provider fixture.

    Implemented architectures are:
    * "MLP":
        - input: 64-dimensional features vectors
        - stack: 32-neurons fully-connected layer with ReLU
                 16-neurons fully-connected layer with ReLU
                 1 output neuron with sigmoid activation
    * "MLP-tune":
        - same as MLP, but freeze the first layer of the stack
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
    framework = "tensorflow"

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
        else:
            raise ValueError("Invalid model 'kind'.")
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
                tf_keras.layers.Dense(32, activation="relu"),
                tf_keras.layers.Dense(16, activation="relu"),
                tf_keras.layers.Dense(1, activation="sigmoid"),
            ]
            shape = [None, 64]
            if self.kind == "MLP-tune":
                stack[0].trainable = False
        elif self.kind == "RNN":
            stack = [
                tf_keras.layers.Embedding(100, 32),
                tf_keras.layers.LSTM(16, activation="tanh"),
                tf_keras.layers.Dense(1, activation="sigmoid"),
            ]
            shape = [None, 128]
        elif self.kind == "CNN":
            cnn_kwargs = {"padding": "same", "activation": "relu"}
            stack = [
                tf_keras.layers.Conv2D(32, 7, **cnn_kwargs),
                tf_keras.layers.MaxPool2D((8, 8)),
                tf_keras.layers.Conv2D(16, 5, **cnn_kwargs),
                tf_keras.layers.AveragePooling2D((8, 8)),
                tf_keras.layers.Reshape((16,)),
                tf_keras.layers.Dense(1, activation="sigmoid"),
            ]
            shape = [None, 64, 64, 3]
        else:
            raise ValueError("Invalid model 'kind'.")
        tfmod = tf_keras.Sequential(stack)
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
    cpu_only: bool,
) -> TensorflowTestCase:
    """Fixture to access a TensorflowTestCase."""
    if cpu_only and (device == "GPU"):
        pytest.skip(reason="--cpu-only mode")
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
        # Set up a model with a frozen layer.
        model = test_case.model
        tfmod = model.get_wrapped_model()
        tfmod.layers[0].trainable = False  # freeze the first layer's weights
        # Access names of the model's variables (via TensorFlow/Keras API).
        if hasattr(tf_keras, "version") and tf_keras.version().startswith("3"):
            names_all_wghts = {v.value.name for v in tfmod.weights}
            names_trainable = {v.value.name for v in tfmod.trainable_weights}
        else:
            names_all_wghts = {v.name for v in tfmod.weights}
            names_trainable = {v.name for v in tfmod.trainable_weights}
        # Verify that DecLearn-accessed weights' names match.
        weights_all = model.get_weights()
        weights_trn = model.get_weights(trainable=True)
        assert set(weights_trn.coefs).issubset(weights_all.coefs)
        assert weights_all.coefs.keys() == names_all_wghts
        assert weights_trn.coefs.keys() == names_trainable

    def test_set_frozen_weights(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that `set_weights` behaves properly with frozen weights."""
        # Setup a model with some frozen weights, and gather trainable ones.
        model = test_case.model
        tfmod = model.get_wrapped_model()
        tfmod.layers[-1].trainable = False  # freeze the last layer's weights
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
        tfmod = model.get_wrapped_model()
        device = f"{test_case.device}:0"
        if hasattr(tf_keras, "version") and tf_keras.version().startswith("3"):
            variables = [var.value for var in tfmod.weights]
        else:
            variables = tfmod.weights
        for var in variables:
            assert var.device.endswith(device)


class TestBuildKerasLoss:
    """Unit tests for `build_keras_loss` util function."""

    def test_build_keras_loss_from_string_class_name(self) -> None:
        """Test `build_keras_loss` with a valid class name string input."""
        loss = build_keras_loss(
            "BinaryCrossentropy", tf_keras.losses.Reduction.SUM
        )
        assert isinstance(loss, tf_keras.losses.BinaryCrossentropy)
        assert loss.reduction == tf_keras.losses.Reduction.SUM

    def test_build_keras_loss_from_string_function_name(self) -> None:
        """Test `build_keras_loss` with a valid function name string input."""
        loss = build_keras_loss(
            "binary_crossentropy", tf_keras.losses.Reduction.SUM
        )
        assert isinstance(loss, tf_keras.losses.BinaryCrossentropy)
        assert loss.reduction == tf_keras.losses.Reduction.SUM

    def test_build_keras_loss_from_string_noclass_function_name(self) -> None:
        """Test `build_keras_loss` with a valid function name string input."""
        if hasattr(tf_keras, "version") and tf_keras.version().startswith("3"):
            pytest.skip("Skipping test that no longer works with Keras 3.")
        loss = build_keras_loss("mse", tf_keras.losses.Reduction.SUM)
        assert isinstance(loss, tf_keras.losses.Loss)
        assert hasattr(loss, "loss_fn")
        assert loss.loss_fn is tf_keras.losses.mse
        assert loss.reduction == tf_keras.losses.Reduction.SUM

    def test_build_keras_loss_from_loss_instance(self) -> None:
        """Test `build_keras_loss` with a valid keras Loss input."""
        # Set up a BinaryCrossentropy loss instance.
        loss = tf_keras.losses.BinaryCrossentropy(
            reduction=tf_keras.losses.Reduction.SUM
        )
        assert loss.reduction == tf_keras.losses.Reduction.SUM
        # Pass it through the util and verify that reduction changes.
        loss = build_keras_loss(loss, tf_keras.losses.Reduction.NONE)
        assert isinstance(loss, tf_keras.losses.BinaryCrossentropy)
        assert loss.reduction == tf_keras.losses.Reduction.NONE

    def test_build_keras_loss_from_loss_function(self) -> None:
        """Test `build_keras_loss` with a valid keras loss function input."""
        loss = build_keras_loss(
            tf_keras.losses.binary_crossentropy, tf_keras.losses.Reduction.SUM
        )
        assert isinstance(loss, tf_keras.losses.Loss)
        assert hasattr(loss, "loss_fn")
        assert loss.loss_fn is tf_keras.losses.binary_crossentropy
        assert loss.reduction == tf_keras.losses.Reduction.SUM

    def test_build_keras_loss_from_custom_function(self) -> None:
        """Test `build_keras_loss` with a valid custom loss function input."""

        def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            """Custom loss function."""
            return tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))

        loss = build_keras_loss(loss_fn, tf_keras.losses.Reduction.SUM)
        assert isinstance(loss, tf_keras.losses.Loss)
        assert hasattr(loss, "loss_fn")
        assert loss.loss_fn is loss_fn
        assert loss.reduction == tf_keras.losses.Reduction.SUM
