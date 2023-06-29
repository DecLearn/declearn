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

"""Unit tests objects for 'declearn.dataset.InMemoryDataset'"""

import sys

import pytest

from declearn.dataset import InMemoryDataset

# Relative imports from the unit tests code of the parent class.
# pylint: disable=wrong-import-order, wrong-import-position
# fmt: off
sys.path.append(".")
from dataset_testbase import DatasetTestSuite, DatasetTestToolbox

sys.path.pop()
# pylint: enable=wrong-import-order, wrong-import-position
# fmt: on


SEED = 0


class InMemoryDatasetTestToolbox(DatasetTestToolbox):
    """Toolbox for InMemoryDataset"""

    # pylint: disable=too-few-public-methods

    framework = "torch"

    def get_dataset(self) -> InMemoryDataset:
        return InMemoryDataset(self.data, self.label, self.weights, seed=SEED)


@pytest.fixture(name="toolbox")
def fixture_dataset() -> DatasetTestToolbox:
    """Fixture to access a InMemoryDatasetTestToolbox."""
    return InMemoryDatasetTestToolbox()


class TestInMemoryDataset(DatasetTestSuite):
    """Unit tests for declearn.dataset.InMemoryDataset."""
