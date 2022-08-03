# coding: utf-8

"""Unit tests for TorchModel."""

import json
from typing import List, Tuple

import numpy as np
import pytest
import torch
from typing_extensions import Literal  # future: import from typing (Py>=3.8)

from declearn2.model.api import NumpyVector
from declearn2.model.torch import TorchModel, TorchVector
from declearn2.typing import Batch
from declearn2.utils import json_pack, json_unpack


class ExtractLSTMFinalOutput(torch.nn.Module):
    """Custom torch Module to gather only the desired output from a LSTM."""

    def forward(
            self,
            inputs: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        ) -> torch.Tensor:
        """Extract the desired Tensor from a LSTM's outputs."""
        return inputs[1][0][0]


class TorchTestCase:
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

    def __init__(
            self,
            kind: Literal["MLP", "RNN", "CNN"],
        ) -> None:
        """Specify the desired model architecture."""
        if kind not in ("MLP", "RNN", "CNN"):
            raise ValueError(f"Invalid torch test architecture: '{kind}'.")
        self.kind = kind

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


@pytest.fixture
def test_case(kind: Literal["MLP", "RNN", "CNN"]) -> TorchTestCase:
    """Fixture to access a TorchTestCase."""
    return TorchTestCase(kind)


@pytest.mark.parametrize("kind", ["MLP", "RNN", "CNN"])
class TestTorchModel:
    """Unit tests for declearn.model.torch.TorchModel."""

    @pytest.mark.xfail  # NOTE: unimplemented feature **yet**
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
        batch = test_case.dataset[0]
        # Check that gradients computation works.
        w_srt = model.get_weights()
        grads = model.compute_batch_gradients(batch)
        w_end = model.get_weights()
        assert w_srt == w_end
        assert isinstance(grads, TorchVector)

    def test_compute_batch_gradients_np(self, test_case):
        """Check that gradients computations work with numpy inputs."""
        # Setup the model and a batch of data, in both tf and np formats.
        model = test_case.model
        nn_batch = test_case.dataset[0]
        assert isinstance(nn_batch[0], torch.Tensor)
        np_batch = [None if arr is None else arr.numpy() for arr in nn_batch]
        assert isinstance(np_batch[0], np.ndarray)
        # Compute gradients in both cases.
        np_grads = model.compute_batch_gradients(np_batch)
        assert isinstance(np_grads, TorchVector)
        nn_grads = model.compute_batch_gradients(nn_batch)
        assert nn_grads == np_grads

    def test_apply_updates(self, test_case):
        """Test that updates' application is mathematically correct."""
        model = test_case.model
        batch = next(iter(test_case.dataset))
        # Compute gradients.
        w_srt = model.get_weights()
        grads = model.compute_batch_gradients(batch)
        # Check that updates can be obtained and applied.
        grads = -.1 * grads
        assert isinstance(grads, TorchVector)
        model.apply_updates(grads)
        # Verify the the updates were correctly applied.
        # Check up to 1e-7 numerical precision due to tf/np conversion.
        # NOTE: if the model has frozen weights, this test would xfail.
        w_end = model.get_weights()
        assert w_end != w_srt
        updt = [val.numpy() for val in grads.coefs.values()]
        diff = list((w_end - w_srt).coefs.values())
        assert all(np.abs(a - b).max() < 1e-6 for a, b in zip(diff, updt))

    def test_serialize_gradients(self, test_case):
        """Test that computed gradients can be (de)serialized as strings."""
        model = test_case.model
        batch = next(iter(test_case.dataset))
        grads = model.compute_batch_gradients(batch)
        gdump = json.dumps(grads.pack(), default=json_pack)
        assert isinstance(gdump, str)
        other = TorchVector.unpack(
            json.loads(gdump, object_hook=json_unpack)
        )
        assert grads == other
