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

"""Unit tests objects for 'declearn.dataset.TorchDataset'"""

import sys
from dataclasses import asdict
from typing import Tuple

import numpy as np
import pytest
import torch

from declearn.dataset import TorchDataset, transform_batch
from declearn.test_utils import assert_batch_equal, to_numpy

# Relative imports from the unit tests code of the parent class.
# pylint: disable=wrong-import-order, wrong-import-position
# fmt: off
sys.path.append(".")
from dataset_testbase import DatasetTestSuite, DatasetTestToolbox

sys.path.pop()
# pylint: enable=wrong-import-order, wrong-import-position
# fmt: on

# false-positives; pylint: disable=no-member

SEED = 0


class CustomDataset(torch.utils.data.Dataset):

    """Basic torch.utils.data.Dataset for testing purposes"""

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

    def get_data_specs(self):
        """Basic get_data_spec method"""
        return {"classes": tuple(range(1, 5))}


class TorchDatasetTestToolbox(DatasetTestToolbox):

    """Toolbox for Torch Dataset"""

    # pylint: disable=too-few-public-methods

    framework = "torch"

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
                to_numpy(res[1], toolbox.framework)[0] == expected[i]
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
                to_numpy(res[1], toolbox.framework)[0] == expected[i]
                for i, res in enumerate(result)
            )
        )

    def test_get_data_specs_custom(self, toolbox: DatasetTestToolbox):
        """Test the get_data_spec method"""
        specs = toolbox.get_dataset().get_data_specs()
        assert asdict(specs)["classes"] == tuple(range(1, 5))

    def test_transform_batch_single_tensor(self, toolbox: DatasetTestToolbox):
        """
        Test the declearn.dataset.transform_batch utility function on
        a single tensor.

        Note : batches are wrapped in list to stick to expected input format
        of torch.utils.data.default_collate(batch)
        """
        batch = [torch.tensor([[1, 2], [3, 4]])]
        data_item_info = {"type": torch.Tensor, "len": 1}
        expected_output = [(np.array([[[1, 2], [3, 4]]]), None, None)]
        output = [transform_batch(batch, data_item_info)]
        assert_batch_equal(output, expected_output, toolbox.framework)

    def test_transform_batch_two_tensors(self, toolbox: DatasetTestToolbox):
        """
        Test the declearn.dataset.transform_batch utility function on
        two tensors tensor.

        Note : batches are wrapped in list to stick to expected input format
        of torch.utils.data.default_collate(batch)
        """
        batch = [
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[1, 2], [3, 4]]),
        ]
        data_item_info = {"type": torch.Tensor, "len": 1}
        expected_output = [
            (np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]), None, None)
        ]
        output = [transform_batch(batch, data_item_info)]
        assert_batch_equal(output, expected_output, toolbox.framework)

    def test_transform_batch_list_in_tuple(self, toolbox: DatasetTestToolbox):
        """
        Test the declearn.dataset.transform_batch utility function on
        a (input,label) tuple, where input is a list of tensors.

        Note : batches are wrapped in list to stick to expected input format
        of torch.utils.data.default_collate(batch)
        """
        batch = [
            (
                [torch.tensor([1, 2]), torch.tensor([3, 4])],
                torch.tensor([0, 0]),
            )
        ]
        data_item_info = {"type": tuple, "len": 2}
        expected_output = [
            (
                [np.array([[1, 2]]), np.array([[3, 4]])],
                torch.tensor([[0, 0]]),
                None,
            )
        ]
        output = [transform_batch(batch, data_item_info)]
        assert_batch_equal(output, expected_output, toolbox.framework)
