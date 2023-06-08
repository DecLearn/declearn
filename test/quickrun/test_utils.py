# coding: utf-8

"""Tests for some 'declearn.quickrun' backend utils."""

import os
import pathlib

import pytest

from declearn.quickrun import parse_data_folder
from declearn.quickrun._parser import (
    get_data_folder_path,
    list_client_names,
)
from declearn.quickrun._run import get_toml_folder


class TestGetTomlFolder:
    """Tests for 'declearn.quickrun._run.get_toml_folder'."""

    def test_get_toml_folder_from_file(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that 'get_toml_folder' works with a TOML file path."""
        config = os.path.join(tmp_path, "config.toml")
        with open(config, "w", encoding="utf-8") as file:
            file.write("")
        toml, folder = get_toml_folder(config)
        assert toml == config
        assert folder == tmp_path.as_posix()

    def test_get_toml_folder_from_folder(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that 'get_toml_folder' works with a folder path."""
        config = os.path.join(tmp_path, "config.toml")
        with open(config, "w", encoding="utf-8") as file:
            file.write("")
        toml, folder = get_toml_folder(tmp_path.as_posix())
        assert toml == config
        assert folder == tmp_path.as_posix()

    def test_get_toml_folder_from_file_fails(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that it fails with a path to a non-existing file."""
        config = os.path.join(tmp_path, "config.toml")
        with pytest.raises(FileNotFoundError):
            get_toml_folder(config)

    def test_get_toml_folder_from_folder_fails(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that it fails with a folder lacking a 'config.toml' file."""
        with pytest.raises(FileNotFoundError):
            get_toml_folder(tmp_path.as_posix())


class TestGetDataFolderPath:
    """Tests for 'declearn.quickrun._parser.get_data_folder_path'."""

    def test_get_data_folder_path_from_data_folder(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that it works with a valid 'data_folder' argument."""
        path = get_data_folder_path(data_folder=tmp_path.as_posix())
        assert isinstance(path, pathlib.Path)
        assert path == tmp_path

    def test_get_data_folder_path_from_root_folder(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that it works with a valid 'root_folder' argument."""
        data_dir = os.path.join(tmp_path, "data")
        os.makedirs(data_dir)
        path = get_data_folder_path(root_folder=tmp_path.as_posix())
        assert isinstance(path, pathlib.Path)
        assert path.as_posix() == data_dir

    def test_get_data_folder_path_fails_no_inputs(
        self,
    ) -> None:
        """Test that it fails with no folder specification."""
        with pytest.raises(ValueError):
            get_data_folder_path(None, None)

    def test_get_data_folder_path_fails_invalid_data_folder(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that it fails with an invalid data_folder."""
        missing_folder = os.path.join(tmp_path, "data")
        with pytest.raises(ValueError):
            get_data_folder_path(data_folder=missing_folder)

    def test_get_data_folder_path_fails_from_root_no_data(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that it fails with an invalid root_folder (no data)."""
        with pytest.raises(ValueError):
            get_data_folder_path(root_folder=tmp_path.as_posix())

    def test_get_data_folder_path_fails_from_root_multiple_data(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that it fails with multiple data* under root_folder."""
        os.makedirs(os.path.join(tmp_path, "data_1"))
        os.makedirs(os.path.join(tmp_path, "data_2"))
        with pytest.raises(ValueError):
            get_data_folder_path(root_folder=tmp_path.as_posix())


class TestListClientNames:
    """Tests for the 'declearn.quickrun._parser.list_client_names' function."""

    def test_list_client_names_from_folder(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that it works with a data folder."""
        os.makedirs(os.path.join(tmp_path, "client_1"))
        os.makedirs(os.path.join(tmp_path, "client_2"))
        names = list_client_names(tmp_path)
        assert isinstance(names, list)
        assert sorted(names) == ["client_1", "client_2"]

    def test_list_client_names_from_names(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that it works with pre-specified names."""
        os.makedirs(os.path.join(tmp_path, "client_1"))
        os.makedirs(os.path.join(tmp_path, "client_2"))
        names = list_client_names(tmp_path, ["client_2"])
        assert names == ["client_2"]

    def test_list_client_names_fails_invalid_names(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Test that it works with invalid pre-specified names."""
        with pytest.raises(ValueError):
            list_client_names(tmp_path, "invalid-type")  # type: ignore
        with pytest.raises(ValueError):
            list_client_names(tmp_path, ["client_2"])
