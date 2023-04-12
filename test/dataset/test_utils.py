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

"""Unit tests for 'declearn.dataset.utils' functions."""

import os

import numpy as np
import pandas as pd  # type: ignore
from scipy.sparse import coo_matrix, csr_matrix  # type: ignore
from sklearn.datasets import dump_svmlight_file  # type: ignore

from declearn.dataset.utils import (
    load_data_array,
    save_data_array,
)


class TestSaveLoadDataArray:
    """Unitary functional tests for data arrays loading and saving utils."""

    def test_save_load_csv(self, tmpdir: str) -> None:
        """Test '(save|load)_data_array' with pandas/csv data."""
        cat = np.random.choice(["a", "b", "c"], size=100)
        num = np.random.normal(size=100).round(6)
        data = pd.DataFrame({"cat": cat, "num": num})
        base = os.path.join(tmpdir, "data")
        # Test that the data can properly be saved.
        path = save_data_array(base, data)
        assert isinstance(path, str)
        assert path.startswith(base) and path.endswith(".csv")
        assert os.path.isfile(path)
        # Test that the data can properly be reloaded.
        dbis = load_data_array(path)
        assert isinstance(dbis, pd.DataFrame)
        assert np.all(data.values == dbis.values)

    def test_save_load_npy(self, tmpdir: str) -> None:
        """Test '(save|load)_data_array' with numpy data."""
        data = np.random.normal(size=(128, 32))
        base = os.path.join(tmpdir, "data")
        # Test that the data can properly be saved.
        path = save_data_array(base, data)
        assert isinstance(path, str)
        assert path.startswith(base) and path.endswith(".npy")
        assert os.path.isfile(path)
        # Test that the data can properly be reloaded.
        dbis = load_data_array(path)
        assert isinstance(dbis, np.ndarray)
        assert np.all(data == dbis)

    def test_save_load_sparse(self, tmpdir: str) -> None:
        """Test '(save|load)_data_array' with sparse data."""
        rng = np.random.default_rng(seed=0)
        val = rng.normal(size=20)
        idx = rng.choice(128, size=20)
        jdx = rng.choice(32, size=20)
        data = coo_matrix((val, (idx, jdx)))
        base = os.path.join(tmpdir, "data")
        # Test that the data can properly be saved.
        path = save_data_array(base, data)
        assert isinstance(path, str)
        assert path.startswith(base) and path.endswith(".sparse")
        assert os.path.isfile(path)
        # Test that the data can properly be reloaded.
        dbis = load_data_array(path)
        assert isinstance(dbis, coo_matrix)
        assert data.shape == dbis.shape
        assert data.nnz == dbis.nnz
        assert np.all(data.toarray() == dbis.toarray())

    def test_load_svmlight(self, tmpdir: str) -> None:
        """Test 'load_data_array' with svmlight data."""
        # Save some data to svmlight using scikit-learn.
        path = os.path.join(tmpdir, "data.svmlight")
        x_dat = np.random.normal(size=(100, 32))
        y_dat = np.random.normal(size=100)
        dump_svmlight_file(x_dat, y_dat, path)
        # Test that the data can properly be reloaded with declearn.
        x_bis = load_data_array(path)
        y_bis = load_data_array(path, which=1)
        assert isinstance(x_bis, csr_matrix)
        assert np.allclose(x_bis.toarray(), x_dat)
        assert isinstance(y_bis, np.ndarray)
        assert np.allclose(y_bis, y_dat)
