import sys

import pytest

from declearn.dataset import InMemoryDataset

sys.path.append(".")
from dataset_testbase import DatasetTestSuite, DatasetTestToolbox

SEED = 0


class InMemoryDatasetTestToolbox(DatasetTestToolbox):
    """Toolbox for InMemoryDataset"""

    # pylint: disable=too-few-public-methods

    framework = "torch"

    def get_dataset(self) -> InMemoryDataset:
        return InMemoryDataset(self.data, self.label, self.weights, seed=SEED)


@pytest.fixture(name="toolbox")
def fixture_dataset() -> DatasetTestToolbox:
    """Fixture to access a TorchTestCase."""
    return InMemoryDatasetTestToolbox()


class TestInMemoryDataset(DatasetTestSuite):
    """Unit tests for declearn.dataset._torch.TorchDataset."""
