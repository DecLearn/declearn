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

"""Unit tests for 'declearn.data_info.ClassesField'."""

import numpy as np
import pytest

from declearn.data_info import ClassesField


class TestClassesField:
    """Unit tests for 'declearn.data_info.ClassesField'."""

    def test_is_valid_list(self) -> None:
        """Test `is_valid` with a valid list value."""
        assert ClassesField.is_valid([0, 1])

    def test_is_valid_set(self) -> None:
        """Test `is_valid` with a valid set value."""
        assert ClassesField.is_valid({0, 1})

    def test_is_valid_tuple(self) -> None:
        """Test `is_valid` with a valid tuple value."""
        assert ClassesField.is_valid((0, 1))

    def test_is_valid_array(self) -> None:
        """Test `is_valid` with a valid numpy array value."""
        assert ClassesField.is_valid(np.array([0, 1]))

    def test_is_invalid_2d_array(self) -> None:
        """Test `is_valid` with an invalid numpy array value."""
        assert not ClassesField.is_valid(np.array([[0, 1], [2, 3]]))

    def test_combine(self) -> None:
        """Test `combine` with valid and compatible inputs."""
        values = ([0, 1], (0, 1), {1, 2}, np.array([1, 3]))
        assert ClassesField.combine(*values) == {0, 1, 2, 3}

    def test_combine_fails(self) -> None:
        """Test `combine` with some invalid inputs."""
        values = ([0, 1], np.array([[0, 1], [2, 3]]))
        with pytest.raises(ValueError):
            ClassesField.combine(*values)
