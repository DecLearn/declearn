"""Standard MNIST CNN built with PyTorch, used by all torch runs."""

import torch

from declearn.model.api import Model
from declearn.model.torch import TorchModel

__all__ = ["build_model"]


def build_model() -> Model:
    """Return a `TorchModel` wrapping a small MNIST CNN.

    Expects inputs of shape `(B, 1, 28, 28)` (the `chw` data layout);
    no `Unflatten` step is included.
    """
    network = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 3, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Dropout(0.25),
        torch.nn.Flatten(),
        torch.nn.Linear(1352, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(64, 10),
        torch.nn.Softmax(dim=-1),
    )
    return TorchModel(network, loss=torch.nn.CrossEntropyLoss())
