# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Functional tests for 'declearn.dataset.examples' utils."""

import os
from unittest import mock

import numpy as np
import pandas as pd  # type: ignore
import pytest
import torch

from declearn.dataset.examples import (
    load_heart_uci,
    load_mnist,
)
from declearn.dataset.examples._time_series_emg import (
    EMGDatasetConfigs,
    load_semg_hand_poses,
)


def test_load_heart_uci(tmpdir: str) -> None:
    """Functional tests for 'declearn.dataset.examples.load_heart_uci'."""
    # Test that downloading the dataset works.
    data, tcol = load_heart_uci("va", folder=tmpdir)
    assert isinstance(data, pd.DataFrame)
    assert tcol in data.columns
    # Test that re-loading the dataset works.
    with mock.patch(
        "declearn.dataset.examples._heart_uci.download_heart_uci"
    ) as patch_download:
        data_bis, tcol_bis = load_heart_uci("va", folder=tmpdir)
        patch_download.assert_not_called()
    assert np.allclose(data.values, data_bis.values)
    assert tcol == tcol_bis


def test_load_mnist(tmpdir: str) -> None:
    """Functional tests for 'declearn.dataset.examples.load_mnist'."""
    # Test that downloading the (test) dataset works.
    images, labels = load_mnist(train=False, folder=tmpdir)
    assert isinstance(images, np.ndarray)
    assert images.shape == (10000, 28, 28)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (images.shape[0],)
    assert (np.unique(labels) == np.arange(10)).all()
    # Test that re-loading the dataset works.
    with mock.patch("requests.get") as patch_download:
        img_bis, lab_bis = load_mnist(train=False, folder=tmpdir)
        patch_download.assert_not_called()
    assert (img_bis == images).all()
    assert (lab_bis == labels).all()


class TestLoadSemgHandPoses:
    """Functional tests for 'load_semg_hand_poses'."""

    def _make_configs(self, path):
        return EMGDatasetConfigs(
            path=path,
            target=1,
            on_save_filename="data_pr",
        )

    def test_download_if_file_not_exists(self, tmp_path):
        configs = self._make_configs(tmp_path)

        data = load_semg_hand_poses(configs)

        assert os.path.exists(f"{tmp_path}/{configs.zip_name}.zip")
        assert isinstance(data, torch.Tensor)
        assert data.dim() >= 2

    def test_no_redownload_if_file_exists(self, tmp_path):
        configs = self._make_configs(tmp_path)

        # first call triggers download
        load_semg_hand_poses(configs)

        # second call should not trigger download logic
        with mock.patch(
            "declearn.dataset.examples._time_series_emg.get_hand_poses_emg_tensor"
        ) as get_hand_poses:
            load_semg_hand_poses(configs)
            get_hand_poses.assert_not_called()

    def test_invalid_path_raises(self):
        configs = self._make_configs("tmp/dir")

        with pytest.raises(ValueError):
            load_semg_hand_poses(configs)
