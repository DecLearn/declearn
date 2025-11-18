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

"""Unit tests for 'declearn.data_info' high-level utils."""

import uuid
from typing import Any, Type
from unittest import mock

import pytest

from declearn.data_info import (
    DataInfoField,
    aggregate_data_info,
    get_data_info_fields_documentation,
    register_data_info_field,
)


class TestAggregateDataInfo:
    """Unit tests for 'declearn.data_info.aggregate_data_info'."""

    def test_aggregate_data_info(self) -> None:
        """Test aggregating valid, compatible data info."""
        clients_data_info = [
            {"n_samples": 10, "features_shape": (100,)},
            {"n_samples": 32, "features_shape": (100,)},
        ]
        result = aggregate_data_info(clients_data_info)
        assert result == {"n_samples": 42, "features_shape": (100,)}

    def test_aggregate_data_info_required(self) -> None:
        """Test aggregating a subset of valid, compatible data info."""
        clients_data_info = [
            {"n_samples": 10, "features_shape": (100,)},
            {"n_samples": 32, "features_shape": (100,)},
        ]
        result = aggregate_data_info(
            clients_data_info, required_fields={"n_samples"}
        )
        assert result == {"n_samples": 42}

    def test_aggregate_data_info_missing_required(self) -> None:
        """Test that a KeyError is raised on missing required data info."""
        clients_data_info = [
            {"n_samples": 10},
            {"n_samples": 32},
        ]
        with pytest.raises(KeyError):
            aggregate_data_info(
                clients_data_info,
                required_fields={"n_samples", "features_shape"},
            )

    def test_aggregate_data_info_invalid_values(self) -> None:
        """Test that a ValueError is raised on invalid values."""
        clients_data_info = [
            {"n_samples": 10},
            {"n_samples": -1},
        ]
        with pytest.raises(ValueError):
            aggregate_data_info(clients_data_info)

    def test_aggregate_data_info_incompatible_values(self) -> None:
        """Test that a ValueError is raised on incompatible values."""
        clients_data_info = [
            {"features_shape": (28,)},
            {"features_shape": (32,)},
        ]
        with pytest.raises(ValueError):
            aggregate_data_info(clients_data_info)

    def test_aggregate_data_info_undefined_field(self) -> None:
        """Test that unspecified fields are handled as expected."""
        clients_data_info = [
            {"n_samples": 10, "undefined": "a"},
            {"n_samples": 32, "undefined": "b"},
        ]
        with mock.patch("warnings.warn") as patch_warn:
            result = aggregate_data_info(clients_data_info)
        patch_warn.assert_called_once()
        assert result == {"n_samples": 42, "undefined": ["a", "b"]}


class TestRegisterDataInfoField:
    """Unit tests for 'declearn.data_info.register_data_info_field'."""

    def create_mock_cls(self) -> Type[DataInfoField]:
        """Create and return a mock DataInfoField subclass."""

        field_name = f"mock_field_{uuid.uuid4()}"

        class MockDataInfoField(DataInfoField):
            """Mock DataInfoField subclass."""

            field = field_name
            types = (str,)
            doc = f"Documentation for '{field_name}'."

            @classmethod
            def combine(cls, *values: Any) -> Any:
                return values

        return MockDataInfoField

    def test_register_data_info_field(self) -> None:
        """Test that registrating a custom DataInfoField works."""
        # Set up a mock DataInfoField subclass.
        mock_cls = self.create_mock_cls()
        # Test that it can be registered, and thereafter accessed.
        register_data_info_field(mock_cls)
        documentation = get_data_info_fields_documentation()
        assert mock_cls.field in documentation
        assert documentation[mock_cls.field] == mock_cls.doc

    def test_register_data_info_field_invalid_type(self) -> None:
        """Test that registering a non-DataInfoField subclass fails."""
        with pytest.raises(TypeError):
            register_data_info_field(int)  # type: ignore

    def test_register_data_info_field_already_used(self) -> None:
        """Test that registering twice under the same name fails."""
        # Set up a couple of DataInfoField mock classes with same field name.
        mock_cls = self.create_mock_cls()
        mock_bis = self.create_mock_cls()
        mock_bis.field = mock_cls.field
        # Test that they cannot both be registered.
        register_data_info_field(mock_cls)
        with pytest.raises(KeyError):
            register_data_info_field(mock_bis)
