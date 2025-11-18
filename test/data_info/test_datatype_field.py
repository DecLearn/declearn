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

"""Unit tests for 'declearn.data_info.DataTypeField'."""

import numpy as np
import pytest

from declearn.data_info import DataTypeField


class TestDataTypeField:
    """Unit tests for 'declearn.data_info.DataTypeField'."""

    def test_is_valid(self) -> None:
        """Test `is_valid` with some valid values."""
        assert DataTypeField.is_valid("float32")
        assert DataTypeField.is_valid("float64")
        assert DataTypeField.is_valid("int32")
        assert DataTypeField.is_valid("uint8")

    def test_is_not_valid(self) -> None:
        """Test `is_valid` with invalid values."""
        assert not DataTypeField.is_valid(np.int32)
        assert not DataTypeField.is_valid("mocktype")

    def test_combine(self) -> None:
        """Test `combine` with valid and compatible inputs."""
        values = ["float32", "float32"]
        assert DataTypeField.combine(*values) == "float32"

    def test_combine_invalid(self) -> None:
        """Test `combine` with invalid inputs."""
        values = ["float32", "mocktype"]
        with pytest.raises(ValueError):
            DataTypeField.combine(*values)

    def test_combine_incompatible(self) -> None:
        """Test `combine` with incompatible inputs."""
        values = ["float32", "float16"]
        with pytest.raises(ValueError):
            DataTypeField.combine(*values)
