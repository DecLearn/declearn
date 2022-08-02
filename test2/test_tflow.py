# coding: utf-8

"""Unit tests for TensorflowModel."""

import json

import numpy as np
import pytest
import tensorflow as tf  # type: ignore
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn2.model.api import NumpyVector
from declearn2.model.tensorflow import TensorflowModel, TensorflowVector


class KerasTestCase:
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

    def __init__(
            self,
            kind: Literal["MLP", "RNN", "CNN"],
        ) -> None:
        """Specify the desired model architecture."""
        if kind not in ("MLP", "RNN", "CNN"):
            raise ValueError(f"Invalid keras test architecture: '{kind}'.")
        self.kind = kind

    @property
    def dataset(
            self,
        ) -> tf.data.Dataset:
        """Suited toy binary-classification dataset."""
        rng = tf.random.get_global_generator()
        rng.reset_from_seed(0)
        if self.kind == "MLP":
            inputs = rng.normal((2, 32, 64))
        elif self.kind == "RNN":
            inputs = rng.uniform((2, 32, 128), 0, 100, tf.int32)
        elif self.kind == "CNN":
            inputs = rng.normal((2, 32, 64, 64, 3))
        labels = rng.uniform((2, 32), 0, 2, tf.int32)
        return tf.data.Dataset.from_tensor_slices((inputs, labels, None))

    @property
    def model(
            self,
        ) -> TensorflowModel:
        """Suited toy binary-classification keras model."""
        if self.kind == "MLP":
            stack = [
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
            shape = [None, 64]
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
        tfmod.build(shape)
        return TensorflowModel(tfmod, loss="binary_crossentropy", metrics=None)


@pytest.fixture
def test_case(kind: Literal["MLP", "RNN", "CNN"]) -> KerasTestCase:
    """Fixture to access a KerasTestCase."""
    return KerasTestCase(kind)


@pytest.mark.parametrize("kind", ["MLP", "RNN", "CNN"])
class TestTensorflowModel:
    """Unit tests for declearn.model.tensorflow.TensorflowModel."""

    def test_serialization(self, test_case):
        """Check that the model can be JSON-(de)serialized properly."""
        model = test_case.model
        config = json.dumps(model.get_config())
        other = model.from_config(json.loads(config))
        assert model.get_config() == other.get_config()

    def test_get_set_weights(self, test_case):
        """Check that weights are properly initialized to zero."""
        model = test_case.model
        w_srt = model.get_weights()
        assert isinstance(w_srt, NumpyVector)
        w_end = w_srt + 1.
        model.set_weights(w_end)
        assert model.get_weights() == w_end

    def test_compute_batch_gradients(self, test_case):
        """Check that gradients computation works."""
        # Setup the model and a batch of data.
        model = test_case.model
        batch = next(iter(test_case.dataset))
        # Check that gradients computation works.
        w_srt = model.get_weights()
        grads = model.compute_batch_gradients(batch)
        w_end = model.get_weights()
        assert w_srt == w_end
        assert isinstance(grads, TensorflowVector)

    def test_compute_batch_gradients_np(self, test_case):
        """Check that gradients computations work with numpy inputs."""
        # Setup the model and a batch of data, in both tf and np formats.
        model = test_case.model
        tf_batch = next(iter(test_case.dataset))
        assert isinstance(tf_batch[0], tf.Tensor)
        np_batch = [None if arr is None else arr.numpy() for arr in tf_batch]
        assert isinstance(np_batch[0], np.ndarray)
        # Compute gradients in both cases.
        np_grads = model.compute_batch_gradients(np_batch)
        assert isinstance(np_grads, TensorflowVector)
        tf_grads = model.compute_batch_gradients(tf_batch)
        assert tf_grads == np_grads

    def test_apply_updates(self, test_case):
        """Test that updates' application is mathematically correct."""
        model = test_case.model
        batch = next(iter(test_case.dataset))
        # Compute gradients.
        w_srt = model.get_weights()
        grads = model.compute_batch_gradients(batch)
        # Check that updates can be obtained and applied.
        grads = -.1 * grads
        assert isinstance(grads, TensorflowVector)
        model.apply_updates(grads)
        # Verify the the updates were correctly applied.
        # Check up to 1e-7 numerical precision due to tf/np conversion.
        # NOTE: if the model has frozen weights, this test would xfail.
        w_end = model.get_weights()
        assert w_end != w_srt
        updt = [val.numpy() for val in grads.coefs.values()]
        diff = list((w_end - w_srt).coefs.values())
        assert all(np.abs(a - b).max() < 1e-7 for a, b in zip(diff, updt))

    def test_compute_loss(self, test_case):
        """Test that loss computation abides by its specs."""
        loss = test_case.model.compute_loss(test_case.dataset)
        assert isinstance(loss, float)

    def test_serialize_gradients(self, test_case):
        """Test that computed gradients can be (de)serialized as strings."""
        model = test_case.model
        batch = next(iter(test_case.dataset))
        grads = model.compute_batch_gradients(batch)
        gdump = grads.serialize()
        assert isinstance(gdump, str)
        other = TensorflowVector.deserialize(gdump)
        assert grads == other
