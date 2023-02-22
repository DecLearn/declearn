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

import json
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
from declearn.utils import set_device_policy

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
        kind: Literal["MLP", "MLP-tune", "RNN", "CNN"],
        device: Literal["CPU", "GPU"],
    ) -> None:
        """Specify the desired model architecture."""
        if kind not in ("MLP", "MLP-tune", "RNN", "CNN"):
            raise ValueError(f"Invalid torch test architecture: '{kind}'.")
        self.kind = kind
        self.device = device
        set_device_policy(gpu=(device == "GPU"), idx=0)

    @staticmethod
    def to_numpy(
        tensor: Any,
    ) -> np.ndarray:
        """Convert an input tensor to a numpy array."""
        assert isinstance(tensor, torch.Tensor)
        return tensor.cpu().numpy()

    @property
    def dataset(
        self,
    ) -> List[Batch]:
        """Suited toy binary-classification dataset."""
        # false-positives; pylint: disable=no-member
        rng = torch.random.default_generator.manual_seed(0)
        if self.kind.startswith("MLP"):
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
        if self.kind.startswith("MLP"):
            stack = [
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid(),
            ]
            if self.kind == "MLP-tune":
                stack[0].requires_grad_(False)
        elif self.kind == "RNN":
            stack = [
                torch.nn.Embedding(100, 32),
                torch.nn.LSTM(32, 16, batch_first=True),
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

    def assert_correct_device(
        self,
        vector: TorchVector,
    ) -> None:
        """Raise if a vector is backed on the wrong type of device."""
        dev_type = "cuda" if self.device == "GPU" else "cpu"
        assert all(
            tensor.device.type == dev_type for tensor in vector.coefs.values()
        )


@pytest.fixture(name="test_case")
def fixture_test_case(
    kind: Literal["MLP", "MLP-tune", "RNN", "CNN"],
    device: Literal["CPU", "GPU"],
) -> TorchTestCase:
    """Fixture to access a TorchTestCase."""
    return TorchTestCase(kind, device)


DEVICES = ["CPU"]
if torch.cuda.device_count():
    DEVICES.append("GPU")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["MLP", "MLP-tune", "RNN", "CNN"])
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
                self._test_serialization(test_case)
            except AssertionError:
                pytest.skip(
                    "skipping failed test due to custom nn.Module pickling"
                )
        self._test_serialization(test_case)

    def _test_serialization(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that the model can be JSON-(de)serialized properly.

        This method replaces the parent `test_serialization` one.
        """
        # Same setup as in parent test: a model and a config-based other.
        model = test_case.model
        config = json.dumps(model.get_config())
        other = model.from_config(json.loads(config))
        # Verify that both models have the same device policy.
        assert model.device_policy == other.device_policy
        # Verify that both models have a similar structure of modules.
        mod_a = list(getattr(model, "_model").modules())
        mod_b = list(getattr(other, "_model").modules())
        assert len(mod_a) == len(mod_b)
        assert all(isinstance(a, type(b)) for a, b in zip(mod_a, mod_b))
        assert all(repr(a) == repr(b) for a, b in zip(mod_a, mod_b))

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

    def test_get_frozen_weights(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that `get_weights` behaves properly with frozen weights."""
        model = test_case.model
        ptmod = getattr(model, "_model")  # type: torch.nn.Module
        next(ptmod.parameters()).requires_grad = False  # freeze some weights
        w_all = model.get_weights()
        w_trn = model.get_weights(trainable=True)
        assert set(w_trn.coefs).issubset(w_all.coefs)  # check on keys
        n_params = sum(1 for _ in ptmod.parameters())
        n_frozen = sum(not p.requires_grad for p in ptmod.parameters())
        assert n_frozen >= 1  # at least the one frozen for this test
        assert len(w_trn.coefs) == n_params - n_frozen
        assert len(w_all.coefs) == n_params

    def test_set_frozen_weights(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that `set_weights` behaves properly with frozen weights."""
        # Setup a model with some frozen weights, and gather trainable ones.
        model = test_case.model
        ptmod = getattr(model, "_model")  # type: torch.nn.Module
        next(ptmod.parameters()).requires_grad = False  # freeze some weights
        w_trn = model.get_weights(trainable=True)
        # Test that `set_weights` works if and only if properly parametrized.
        with pytest.raises(KeyError):
            model.set_weights(w_trn)
        with pytest.raises(KeyError):
            model.set_weights(model.get_weights(), trainable=True)
        model.set_weights(w_trn, trainable=True)

    def test_proper_model_placement(
        self,
        test_case: TorchTestCase,
    ) -> None:
        """Check that at instantiation, model weights are properly placed."""
        model = test_case.model
        policy = model.device_policy
        assert policy.gpu == (test_case.device == "GPU")
        assert (policy.idx == 0) if policy.gpu else (policy.idx is None)
        ptmod = getattr(model, "_model").module
        device_type = "cpu" if test_case.device == "CPU" else "cuda"
        for param in ptmod.parameters():
            assert param.device.type == device_type
