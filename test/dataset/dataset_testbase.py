from abc import abstractmethod
from typing import Any, Tuple

import numpy as np
from numpy.testing import assert_array_equal

from declearn.dataset import Dataset, DataSpecs


class DatasetTestToolbox:
    """TestCase fixture-provider protocol."""

    data, label, weights = (
        np.concatenate([np.ones((2, 2)), np.zeros((2, 2))]),
        np.array([1, 2, 3, 4]),
        np.ones(4),
    )

    @abstractmethod
    def get_dataset(self) -> Dataset:
        """Convert the test data into a framework-specific dataset"""

    @staticmethod
    def to_numpy(
        batch: Tuple[Any],
    ) -> Tuple[np.ndarray]:
        """Convert an input tensor to a numpy array."""

    def assert_result(self, result, expected):
        for i, res in enumerate(result):
            res = self.to_numpy(res)
            for j, el in enumerate(expected[i]):
                assert_array_equal(res[j], el)


class DatasetTestSuite:

    """Base tests for declearn Dataset abstract methods"""

    def test_generate_batches_batchsize(self, toolbox: DatasetTestToolbox):
        """Test batch_size argument to test_generate_batches method"""
        expected = [
            (np.ones((2, 2)), np.array([1, 2]), np.array([1.0, 1.0])),
            (np.zeros((2, 2)), np.array([3, 4]), np.array([1.0, 1.0])),
        ]
        result = toolbox.get_dataset().generate_batches(2)
        toolbox.assert_result(result, expected)

    def test_generate_batches_shuffle(self, toolbox: DatasetTestToolbox):
        """Test the shuffle argument of the generate_batches method"""
        excluded = range(1, 5)
        result = toolbox.get_dataset().generate_batches(1, shuffle=True)
        assert any(
            (
                toolbox.to_numpy(res)[1][0] != excluded[i]
                for i, res in enumerate(result)
            )
        )

    def test_generate_batches_remainder(self, toolbox: DatasetTestToolbox):
        """Test the drop_remainder argument of the generate_batches method"""
        # drop_remainder = True
        expected = [
            (
                np.concatenate([np.ones((2, 2)), np.zeros((1, 2))]),
                np.array([1, 2, 3]),
                np.ones(3),
            )
        ]
        result = toolbox.get_dataset().generate_batches(3)
        toolbox.assert_result(result, expected)
        # drop_remainder = False
        expected = [
            (
                np.concatenate([np.ones((2, 2)), np.zeros((1, 2))]),
                np.array([1, 2, 3]),
                np.ones(3),
            ),
            (np.array([[0, 0]]), np.array([4]), np.array([1.0])),
        ]
        result = toolbox.get_dataset().generate_batches(
            3, drop_remainder=False
        )
        toolbox.assert_result(result, expected)

    def test_get_data_specs(self, toolbox: DatasetTestToolbox):
        """Test the get_data_spec method"""
        assert toolbox.get_dataset().get_data_specs() == DataSpecs(n_samples=4)
