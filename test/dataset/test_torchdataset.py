import sys
from typing import Any, Tuple

import numpy as np
import pytest
import torch

from declearn.dataset import TorchDataset

sys.path.append(".")
from dataset_testbase import DatasetTestSuite, DatasetTestToolbox

SEED = 0


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, weights) -> None:
        self.inputs = inputs
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx, :], self.labels[idx], self.weights[idx]


class TorchDatasetTestToolbox(DatasetTestToolbox):
    def __init__(self):
        pass

    @staticmethod
    def to_numpy(
        batch: Tuple[torch.Tensor],
    ) -> Tuple[np.ndarray]:
        """Convert an input tensor to a numpy array."""
        return tuple(map(lambda x: x.cpu().numpy(), batch))

    def get_dataset(self) -> TorchDataset:
        return TorchDataset(
            CustomDataset(self.data, self.label, self.weights), seed=SEED
        )


@pytest.fixture(name="toolbox")
def fixture_dataset() -> DatasetTestToolbox:
    """Fixture to access a TorchTestCase."""
    return TorchDatasetTestToolbox()


class TestTorchDataset(DatasetTestSuite):
    """Unit tests for declearn.dataset._torch.TorchDataset."""

    def test_generate_batches_shuffle_seeded(
        self, toolbox: DatasetTestToolbox
    ):
        """Test the shuffle argument of the generate_batches method
        Note: imperfect test, depends on a specific seed implementation"""
        expected = np.array([1, 2, 4, 3])
        result = toolbox.get_dataset().generate_batches(1, shuffle=True)
        assert all(
            (
                toolbox.to_numpy(res)[1][0] == expected[i]
                for i, res in enumerate(result)
            )
        )

    def test_generate_batches_replacement_seeded(
        self, toolbox: DatasetTestToolbox
    ):
        """Test the replacement argument of the generate_batches method
        Note: imperfect test, depends on a specific seed implementation"""
        expected = np.array([1, 4, 2, 1])
        result = toolbox.get_dataset().generate_batches(
            1, replacement=True, shuffle=True
        )
        assert all(
            (
                toolbox.to_numpy(res)[1][0] == expected[i]
                for i, res in enumerate(result)
            )
        )
