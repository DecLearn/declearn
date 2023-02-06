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
        kind: Literal["MLP", "RNN", "CNN"],
    ) -> None:
        """Specify the desired model architecture."""
        if kind not in ("MLP", "RNN", "CNN"):
            raise ValueError(f"Invalid keras test architecture: '{kind}'.")
        self.kind = kind

    @staticmethod
    def to_numpy(
        tensor: Any,
    ) -> np.ndarray:
        """Convert an input tensor to a numpy array."""
        assert isinstance(tensor, tf.Tensor)
        return tensor.numpy()  # type: ignore

    @property
    def dataset(
        self,
    ) -> List[Batch]:
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
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, None))
        return list(iter(dataset))

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
        tfmod.build(shape)  # as model is built, no data_info is required
        return TensorflowModel(tfmod, loss="binary_crossentropy", metrics=None)


@pytest.fixture(name="test_case")
def fixture_test_case(
    kind: Literal["MLP", "RNN", "CNN"]
) -> TensorflowTestCase:
    """Fixture to access a TensorflowTestCase."""
    return TensorflowTestCase(kind)


@pytest.mark.parametrize("kind", ["MLP", "RNN", "CNN"])
@pytest.mark.filterwarnings("ignore:.*randrange.*:DeprecationWarning")
class TestTensorflowModel(ModelTestSuite):
    """Unit tests for declearn.model.tensorflow.TensorflowModel."""
