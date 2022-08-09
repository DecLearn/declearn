# coding: utf-8

"""Unit tests for TorchModel."""

import sys
from typing import Any, List, Tuple

import numpy as np
import pytest
import torch
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn2.model.torch import TorchModel, TorchVector
from declearn2.typing import Batch

# dirty trick to import from `model_testing.py`;
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append('.')
from model_testing import ModelTestSuite, ModelTestCase


class ExtractLSTMFinalOutput(torch.nn.Module):
    """Custom torch Module to gather only the desired output from a LSTM."""

    def forward(
            self,
            inputs: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        ) -> torch.Tensor:
        """Extract the desired Tensor from a LSTM's outputs."""
        return inputs[1][0][0]


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
                torch.nn.Flatten(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid(),
            ]
        nnmod = torch.nn.Sequential(*stack)
        return TorchModel(nnmod, loss=torch.nn.BCELoss())


@pytest.fixture(name="test_case")
def fixture_test_case(
        kind: Literal["MLP", "RNN", "CNN"]
    ) -> TorchTestCase:
    """Fixture to access a TorchTestCase."""
    return TorchTestCase(kind)


@pytest.mark.parametrize("kind", ["MLP", "RNN", "CNN"])
class TestTorchModel(ModelTestSuite):
    """Unit tests for declearn.model.torch.TorchModel."""

    @pytest.mark.xfail  # NOTE: unimplemented feature **yet**
    def test_serialization(
            self,
            test_case: ModelTestCase,
        ) -> None:
        super().test_serialization(test_case)