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

"""Unit tests for TorchModel."""

import sys
from typing import Any, List, Literal, Tuple

import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    pytest.skip("PyTorch is unavailable", allow_module_level=True)

from declearn.model.torch import TorchModel, TorchVector
from declearn.typing import Batch

# dirty trick to import from `model_testing.py`;
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(".")
from model_testing import ModelTestSuite, ModelTestCase


class ExtractLSTMFinalOutput(torch.nn.Module):
    """Custom torch Module to gather only the desired output from a LSTM."""

    def forward(
        self,
        inputs: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Extract the desired Tensor from a LSTM's outputs."""
        return inputs[1][0][0]


class FlattenCNNOutput(torch.nn.Module):
    """Custom torch Module to reshape the output of the CNN conv layers."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reshape the Tensor to the desired shape."""
        shape = (-1,) if (inputs.ndim == 3) else (inputs.shape[0], -1)
        return inputs.view(*shape)


class TorchTestCase(ModelTestCase):
    """PyTorch test-case-provider fixture.

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

    vector_cls = TorchVector
    tensor_cls = torch.Tensor

    def __init__(
        self,
        kind: Literal["MLP", "RNN", "CNN"],
    ) -> None:
        """Specify the desired model architecture."""
        if kind not in ("MLP", "RNN", "CNN"):
            raise ValueError(f"Invalid torch test architecture: '{kind}'.")
        self.kind = kind

    @staticmethod
    def to_numpy(
        tensor: Any,
    ) -> np.ndarray:
        """Convert an input tensor to a numpy array."""
        assert isinstance(tensor, torch.Tensor)
        return tensor.numpy()  # type: ignore

    @property
    def dataset(
        self,
    ) -> List[Batch]:
        """Suited toy binary-classification dataset."""
        # false-positives; pylint: disable=no-member
        rng = torch.random.default_generator.manual_seed(0)
        if self.kind == "MLP":
            inputs = torch.randn((2, 32, 64), generator=rng)
        elif self.kind == "RNN":
            inputs = torch.randint(0, 100, (2, 32, 128), generator=rng)
        elif self.kind == "CNN":
            inputs = torch.randn((2, 32, 3, 64, 64), generator=rng)
        labels = torch.randint(0, 2, (2, 32, 1), generator=rng)
        labels = labels.type(torch.float)
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        return [(*batch, None) for batch in dataset]  # type: ignore

    @property
    def model(
        self,
    ) -> TorchModel:
        """Suited toy binary-classification torch model."""
        if self.kind == "MLP":
            stack = [
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid(),
            ]
        elif self.kind == "RNN":
            stack = [
                torch.nn.Embedding(100, 32),
                torch.nn.LSTM(32, 16, batch_first=True),  # type: ignore
                ExtractLSTMFinalOutput(),
                torch.nn.Tanh(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid(),
            ]
        elif self.kind == "CNN":
            stack = [
                torch.nn.Conv2d(3, 32, 7, padding="same"),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(8),
                torch.nn.Conv2d(32, 16, 5, padding="same"),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(8),
                FlattenCNNOutput(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid(),
            ]
        nnmod = torch.nn.Sequential(*stack)
        return TorchModel(nnmod, loss=torch.nn.BCELoss())


@pytest.fixture(name="test_case")
def fixture_test_case(kind: Literal["MLP", "RNN", "CNN"]) -> TorchTestCase:
    """Fixture to access a TorchTestCase."""
    return TorchTestCase(kind)


@pytest.mark.parametrize("kind", ["MLP", "RNN", "CNN"])
class TestTorchModel(ModelTestSuite):
    """Unit tests for declearn.model.torch.TorchModel."""

    @pytest.mark.filterwarnings("ignore: PyTorch JSON serialization")
    def test_serialization(
        self,
        test_case: ModelTestCase,
    ) -> None:
        if getattr(test_case, "kind", "") == "RNN":
            # NOTE: this test fails on python 3.8 but succeeds in 3.10
            #       due to the (de)serialization of a custom nn.Module
            #       the expected model behaviour is, however, correct
            try:
                super().test_serialization(test_case)
            except AssertionError:
                pytest.skip(
                    "skipping failed test due to custom nn.Module pickling"
                )
        super().test_serialization(test_case)

    def test_compute_batch_gradients_clipped(
        self,
        test_case: ModelTestCase,
    ) -> None:
        if getattr(test_case, "kind", "") == "RNN":
            try:
                super().test_compute_batch_gradients_clipped(test_case)
            except RuntimeError:
                pytest.skip(
                    "skipping test due to lack of RNN support in functorch"
                )
        else:
            super().test_compute_batch_gradients_clipped(test_case)
