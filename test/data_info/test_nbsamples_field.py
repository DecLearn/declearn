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

"""Unit tests for 'declearn.data_info.NbSamplesField'."""

import pytest

from declearn.data_info import NbSamplesField


class TestNbSamplesField:
    """Unit tests for 'declearn.data_info.NbSamplesField'."""

    def test_is_valid(self) -> None:
        """Test `is_valid` with some valid input values."""
        assert NbSamplesField.is_valid(32)
        assert NbSamplesField.is_valid(100)
        assert NbSamplesField.is_valid(8192)

    def test_is_not_valid(self) -> None:
        """Test `is_valid` with invalid values."""
        assert not NbSamplesField.is_valid(16.5)
        assert not NbSamplesField.is_valid(-12)
        assert not NbSamplesField.is_valid(None)

    def test_combine(self) -> None:
        """Test `combine` with valid and compatible inputs."""
        values = [32, 128]
        assert NbSamplesField.combine(*values) == 160
        values = [64, 64, 64, 64]
        assert NbSamplesField.combine(*values) == 256

    def test_combine_invalid(self) -> None:
        """Test `combine` with invalid inputs."""
        values = [128, -12]
        with pytest.raises(ValueError):
            NbSamplesField.combine(*values)
