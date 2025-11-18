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

"""Unit tests for 'declearn.data_info.FeaturesShapeField'."""

import pytest

from declearn.data_info import FeaturesShapeField


class TestFeaturesShapeField:
    """Unit tests for 'declearn.data_info.FeaturesShapeField'."""

    def test_is_valid(self) -> None:
        """Test `is_valid` with some valid input values."""
        # 1-d ; fixed 3-d (image-like) ; variable 2-d (text-like).
        assert FeaturesShapeField.is_valid([32])
        assert FeaturesShapeField.is_valid([64, 64, 3])
        assert FeaturesShapeField.is_valid([None, 128])
        # Same inputs, as tuples.
        assert FeaturesShapeField.is_valid((32,))
        assert FeaturesShapeField.is_valid((64, 64, 3))
        assert FeaturesShapeField.is_valid((None, 128))

    def test_is_not_valid(self) -> None:
        """Test `is_valid` with invalid values."""
        assert not FeaturesShapeField.is_valid(32)
        assert not FeaturesShapeField.is_valid([32, -1])

    def test_combine(self) -> None:
        """Test `combine` with valid and compatible inputs."""
        # 1-d inputs.
        values = [[32], (32,)]
        assert FeaturesShapeField.combine(*values) == (32,)
        # 3-d fixed-size inputs.
        values = [[16, 16, 3], (16, 16, 3)]
        assert FeaturesShapeField.combine(*values) == (16, 16, 3)
        # 2-d variable-size inputs.
        values = [[None, 512], (None, 512)]  # type: ignore
        assert FeaturesShapeField.combine(*values) == (None, 512)

    def test_combine_invalid(self) -> None:
        """Test `combine` with invalid inputs."""
        values = [[32], [32, -1]]
        with pytest.raises(ValueError):
            FeaturesShapeField.combine(*values)

    def test_combine_incompatible(self) -> None:
        """Test `combine` with incompatible inputs."""
        values = [(None, 32), (128,)]
        with pytest.raises(ValueError):
            FeaturesShapeField.combine(*values)
