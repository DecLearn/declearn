# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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

import dataclasses
import os
from typing import List, Tuple, Union

import numpy as np
import pytest

# pylint: disable=duplicate-code
try:
    import torch
except ModuleNotFoundError:
    pytest.skip("PyTorch is unavailable", allow_module_level=True)
# pylint: enable=duplicate-code

from declearn.dataset.torch import TorchDataset, collate_with_padding
from declearn.test_utils import assert_batch_equal, make_importable, to_numpy

# relative imports from `dataset_testbase.py`
with make_importable(os.path.dirname(__file__)):
    from dataset_testbase import DatasetTestSuite, DatasetTestToolbox


SEED = 0


class CustomDataset(torch.utils.data.Dataset):
    """Basic torch.utils.data.Dataset for testing purposes."""

    def __init__(self, inputs, labels, weights) -> None:
        self.inputs = inputs
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx], self.weights[idx]

    def get_data_specs(self):
        """Basic get_data_spec method"""
        return {"classes": tuple(range(1, 5))}


class TorchDatasetTestToolbox(DatasetTestToolbox):
    """Toolbox for Torch Dataset."""

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


class SentencesDataset(torch.utils.data.Dataset):
    """Custom torch.utils.data.Dataset with sequences of tokens as inputs."""

    def __init__(self) -> None:
        # Generate 32 variable-size sequences of int (tokenized sentences).
        rng = np.random.default_rng(seed=SEED)
        self.inputs = torch.utils.data.default_convert(
            [rng.choice(128, size=rng.choice(64) + 1) for _ in range(32)]
        )
        # Generate scalar binary labels.
        self.labels = torch.from_numpy(  # pylint: disable=no-member
            rng.choice(2, size=32)
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


@pytest.fixture(name="sentences_dataset")
def fixture_sentences_dataset() -> torch.utils.data.Dataset:
    """Fixture to setup a torch Dataset yielding sequences of tokens."""
    return SentencesDataset()


class TestTorchDataset(DatasetTestSuite):
    """Unit tests for declearn.dataset._torch.TorchDataset."""

    def test_generate_batches_shuffle_seeded(
        self,
        toolbox: TorchDatasetTestToolbox,
    ) -> None:
        """Test the shuffle argument of the generate_batches method
        Note: imperfect test, depends on a specific seed implementation"""
        expected = np.array([1, 2, 4, 3])
        result = toolbox.get_dataset().generate_batches(1, shuffle=True)
        framework = toolbox.framework
        assert all(
            to_numpy(res[1], framework)[0] == expected[i]
            for i, res in enumerate(result)
        )

    def test_generate_batches_replacement_seeded(
        self,
        toolbox: TorchDatasetTestToolbox,
    ) -> None:
        """Test the replacement argument of the generate_batches method
        Note: imperfect test, depends on a specific seed implementation"""
        expected = np.array([1, 4, 2, 1])
        result = toolbox.get_dataset().generate_batches(
            1, replacement=True, shuffle=True
        )
        framework = toolbox.framework
        assert all(
            to_numpy(res[1], framework)[0] == expected[i]
            for i, res in enumerate(result)
        )

    def test_get_data_specs_custom(
        self,
        toolbox: TorchDatasetTestToolbox,
    ) -> None:
        """Test the get_data_spec method"""
        specs = toolbox.get_dataset().get_data_specs()
        assert dataclasses.asdict(specs)["classes"] == tuple(range(1, 5))

    def test_collate_to_batch_single_tensor(
        self,
        toolbox: TorchDatasetTestToolbox,
    ) -> None:
        """Test the default 'collate_to_batch' with single-tensor x samples."""
        samples: List[Union[torch.Tensor, List[torch.Tensor]]] = [
            torch.Tensor([1, 2]),
            torch.Tensor([3, 4]),
        ]
        expected_output = (
            torch.Tensor([[1, 2], [3, 4]]),
            None,
            None,
        )
        dataset = toolbox.get_dataset()
        output = dataset.collate_to_batch(samples)
        assert_batch_equal(output, expected_output, toolbox.framework)

    def test_collate_to_batch_single_tensor_in_tuple(
        self,
        toolbox: TorchDatasetTestToolbox,
    ) -> None:
        """Test the default 'collate_to_batch' with (x,) samples."""
        samples: List[Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]] = [
            (torch.Tensor([1, 2]),),
            (torch.Tensor([3, 4]),),
        ]
        expected_output = (
            torch.Tensor([[1, 2], [3, 4]]),
            None,
            None,
        )
        dataset = toolbox.get_dataset()
        output = dataset.collate_to_batch(samples)
        assert_batch_equal(output, expected_output, toolbox.framework)

    def test_collate_to_batch_two_tensors(
        self,
        toolbox: TorchDatasetTestToolbox,
    ) -> None:
        """Test the default 'collate_to_batch' with (x, y) samples."""
        samples: List[Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]] = [
            (torch.Tensor([1, 2]), torch.Tensor([0.0])),
            (torch.Tensor([3, 4]), torch.Tensor([1.0])),
        ]
        expected_output = (
            torch.Tensor([[1, 2], [3, 4]]),
            torch.Tensor([[0.0], [1.0]]),
            None,
        )
        dataset = toolbox.get_dataset()
        output = dataset.collate_to_batch(samples)
        assert_batch_equal(output, expected_output, toolbox.framework)

    def test_collate_to_batch_list_in_tuple(
        self,
        toolbox: TorchDatasetTestToolbox,
    ) -> None:
        """Test the default 'collate_to_batch' with ([x1, x2], y) samples."""
        samples: List[Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]] = [
            ([torch.Tensor([1, 2]), torch.Tensor([3, 4])], torch.Tensor([0])),
            ([torch.Tensor([5, 6]), torch.Tensor([7, 8])], torch.Tensor([1])),
        ]
        expected_output = (
            [torch.Tensor([[1, 2], [5, 6]]), torch.Tensor([[3, 4], [7, 8]])],
            torch.Tensor([[0], [1]]),
            None,
        )
        dataset = toolbox.get_dataset()
        output = dataset.collate_to_batch(samples)
        assert_batch_equal(output, expected_output, toolbox.framework)

    def test_collate_to_batch_multiple_inputs_no_labels(
        self,
        toolbox: TorchDatasetTestToolbox,
    ) -> None:
        """Test the default 'collate_to_batch' with [x1, x2] samples."""
        samples: List[Union[torch.Tensor, List[torch.Tensor]]] = [
            [torch.Tensor([1, 2]), torch.Tensor([3, 4])],
            [torch.Tensor([5, 6]), torch.Tensor([7, 8])],
        ]
        expected_output = (
            [torch.Tensor([[1, 2], [5, 6]]), torch.Tensor([[3, 4], [7, 8]])],
            None,
            None,
        )
        dataset = toolbox.get_dataset()
        output = dataset.collate_to_batch(samples)
        assert_batch_equal(output, expected_output, toolbox.framework)

    def test_generate_padded_batches(
        self,
        sentences_dataset: torch.utils.data.Dataset,
    ) -> None:
        """Test 'generate_batches' with samples that require padding."""
        dataset = TorchDataset(
            sentences_dataset, collate_fn=collate_with_padding, seed=SEED
        )
        batches = list(dataset.generate_batches(batch_size=4, poisson=False))
        # Verify that there are 8 batches, with inputs of shape (4, [1-64]).
        assert len(batches) == 8
        shapes = [inputs.shape for inputs, _, _ in batches]  # type: ignore
        assert all(shp[0] == 4 for shp in shapes)  # single batch size
        assert len(set(shp[1] for shp in shapes)) > 1  # various seq length
        assert all(1 <= shp[1] <= 64 for shp in shapes)

    def test_generate_padded_batches_with_poisson(
        self,
        sentences_dataset: torch.utils.data.Dataset,
    ) -> None:
        """Test 'generate_batches(poisson=True)' with samples to be padded."""
        dataset = TorchDataset(
            sentences_dataset, collate_fn=collate_with_padding, seed=SEED
        )
        batches = list(dataset.generate_batches(batch_size=4, poisson=True))
        # Verify that there are 8 batches, with inputs of shape (?, [1-64]).
        assert len(batches) == 8
        shapes = [inputs.shape for inputs, _, _ in batches]  # type: ignore
        assert len(set(shp[0] for shp in shapes)) > 1  # various batch size
        assert len(set(shp[1] for shp in shapes)) > 1  # various seq length
        assert all(1 <= shp[1] <= 64 for shp in shapes)
