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
from typing import Any, Iterator
from unittest import mock

import pytest

from declearn.dataset._base import Dataset
from declearn.main.utils._error import InvalidDatasetType
from declearn.main.utils._multi_model import (
    dispatch_replies,
    validate_optional_dataset_list,
)
from declearn.messaging._base import MultiModelTrainReply, TrainReply
from declearn.typing import Batch


class DummyDataset(Dataset):
    """Minimal concrete Dataset subclass for tests."""

    typekey = "dummy"

    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError

    def generate_batches(self, batch_size: int = 8) -> Iterator[Batch]:
        # For tests, you can yield no batches or dummy batches
        return iter([])  # or yield dummy Batch objects if needed

    def get_data_specs(self) -> Any:  # replace Any with the real return type
        # Return minimal specs sufficient for tests
        return {}


class TestValidateOptionalDatasetList:
    """Tests different scenarios for dataset validation."""

    def test_data_none_with_reference_length(self) -> None:
        """Tests when the case where the dataset is empty"""
        result = validate_optional_dataset_list(
            data=None,
            name="datasets",
            reference_length=3,
        )
        assert result == [None, None, None]

    def test_data_none_without_reference_length_raises(self) -> None:
        """Tests when the data is None but no reference length is provided"""
        with pytest.raises(
            ValueError,
            match="datasets is None and no reference_length provided",
        ):
            validate_optional_dataset_list(
                data=None,
                name="datasets",
                reference_length=None,
            )

    def test_data_empty_list_no_reference(self) -> None:
        result = validate_optional_dataset_list(
            data=[],
            name="datasets",
            reference_length=None,
        )
        assert result == []

    def test_data_list_of_datasets_no_reference(self) -> None:
        """Tests with the data is a list with no reference length"""
        ds1 = DummyDataset(1)
        ds2 = DummyDataset(2)
        result = validate_optional_dataset_list(
            data=[ds1, ds2],
            name="datasets",
            reference_length=None,
        )
        assert result == [ds1, ds2]

    def test_data_list_with_none_elements(self) -> None:
        """Tests when there exist one None element in the dataset array"""
        ds = DummyDataset(1)
        result = validate_optional_dataset_list(
            data=[ds, None, ds],
            name="datasets",
            reference_length=None,
        )
        assert result == [ds, None, ds]

    def test_reference_length_matches(self) -> None:
        """Tests for dataset length and reference length match"""
        ds1 = DummyDataset(1)
        result = validate_optional_dataset_list(
            data=[ds1, None],
            name="datasets",
            reference_length=2,
        )
        assert result == [ds1, None]

    def test_reference_length_mismatch_raises(self) -> None:
        """Tests for dataset length and reference length mismatch"""
        ds1 = DummyDataset(1)
        with pytest.raises(ValueError, match="datasets must have length 2"):
            validate_optional_dataset_list(
                data=[ds1],
                name="datasets",
                reference_length=2,
            )

    def test_invalid_element_type_raises(self) -> None:
        """Tests wether an error is raised if one of the elements
        is an invalid dataset."""
        ds = DummyDataset(1)
        with pytest.raises(InvalidDatasetType) as exc_info:
            validate_optional_dataset_list(
                data=[ds, "not_a_dataset"],  # type: ignore
                name="datasets",
                reference_length=None,
            )

        assert "datasets[1]" in str(exc_info.value)
        assert "str" in str(exc_info.value)

    def test_invalid_element_type_with_int(self) -> None:
        ds = DummyDataset(1)
        with pytest.raises(InvalidDatasetType, match="datasets\\[1\\]"):
            validate_optional_dataset_list(
                data=[ds, 123],  # type: ignore
                name="datasets",
                reference_length=None,
            )


class TestMultiModelUtils:
    """Tests different utilities for handling multi-model case."""

    @staticmethod
    def make_train_reply(name: str) -> TrainReply:
        """Create a minimal TrainReply for testing."""
        return TrainReply(
            n_epoch=1,
            n_steps=10,
            t_spent=0.5,
            updates=mock.Mock(name=f"{name}_updates"),
            aux_var={},
        )

    def test_dispatch_replies(
        self,
    ):
        """Test dispatching multi-model replies by model index."""
        client_0_model_0 = self.make_train_reply("client_0_model_0")
        client_0_model_1 = self.make_train_reply("client_0_model_1")

        client_1_model_0 = self.make_train_reply("client_1_model_0")
        client_1_model_1 = self.make_train_reply("client_1_model_1")

        client_2_model_0 = self.make_train_reply("client_2_model_0")
        client_2_model_1 = self.make_train_reply("client_2_model_1")
        results = {
            "client_0": MultiModelTrainReply(
                replies=[
                    client_0_model_0,
                    client_0_model_1,
                ]
            ),
            "client_1": MultiModelTrainReply(
                replies=[
                    client_1_model_0,
                    client_1_model_1,
                ]
            ),
            "client_2": MultiModelTrainReply(
                replies=[
                    client_2_model_0,
                    client_2_model_1,
                ]
            ),
        }

        results_per_model = dispatch_replies(
            results,
            lambda reply: reply.replies,
        )
        # Two models -> two dictionaries.
        assert len(results_per_model) == 2
        # Model 0 receives the first reply from every client.
        assert results_per_model[0] == {
            "client_0": client_0_model_0,
            "client_1": client_1_model_0,
            "client_2": client_2_model_0,
        }
        # Model 1 receives the second reply from every client.
        assert results_per_model[1] == {
            "client_0": client_0_model_1,
            "client_1": client_1_model_1,
            "client_2": client_2_model_1,
        }

    def test_dispatch_replies_with_different_number_of_models(
        self,
    ):
        """Test that inconsistent client replies are rejected."""
        results = {
            "client_0": MultiModelTrainReply(
                replies=[
                    self.make_train_reply("client_0_model_0"),
                    self.make_train_reply("client_0_model_1"),
                ]
            ),
            "client_1": MultiModelTrainReply(
                replies=[
                    self.make_train_reply("client_1_model_0"),
                ]
            ),
        }
        with pytest.raises(
            ValueError,
            match="All clients must provide the same number of replies",
        ):
            dispatch_replies(
                results,
                lambda reply: reply.replies,
            )

    def test_dispatch_replies_empty(
        self,
    ):
        """Test dispatching an empty result."""
        results = {}

        results_per_model = dispatch_replies(
            results,
            lambda reply: reply.replies,
        )

        assert results_per_model == []
