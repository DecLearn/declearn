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

"""Unit tests for 'declearn.dataset.InMemoryDataset'"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse  # type: ignore
import sklearn.datasets  # type: ignore

from declearn.dataset import InMemoryDataset
from declearn.dataset.utils import save_data_array
from declearn.test_utils import make_importable

# relative imports from `dataset_testbase.py`
with make_importable(os.path.dirname(__file__)):
    from dataset_testbase import DatasetTestSuite, DatasetTestToolbox


SEED = 0


### Shared-tests-based tests, revolving around batches generation.


class InMemoryDatasetTestToolbox(DatasetTestToolbox):
    """Toolbox for InMemoryDataset"""

    # pylint: disable=too-few-public-methods

    framework = "numpy"

    def get_dataset(self) -> InMemoryDataset:
        return InMemoryDataset(self.data, self.label, self.weights, seed=SEED)


@pytest.fixture(name="toolbox")
def fixture_toolbox() -> DatasetTestToolbox:
    """Fixture to access a InMemoryDatasetTestToolbox."""
    return InMemoryDatasetTestToolbox()


class TestInMemoryDataset(DatasetTestSuite):
    """Unit tests for declearn.dataset.InMemoryDataset."""


### InMemoryDataset-specific unit tests.


@pytest.fixture(name="dataset")
def dataset_fixture() -> pd.DataFrame:
    """Fixture providing with a small toy dataset."""
    rng = np.random.default_rng(seed=SEED)
    wgt = rng.normal(size=10).astype("float32")
    data = {
        "col_a": np.arange(10, dtype="float32"),
        "col_b": rng.normal(size=10).astype("float32"),
        "col_y": rng.choice(3, size=10, replace=True),
        "col_w": wgt / sum(wgt),
    }
    return pd.DataFrame(data)


class TestInMemoryDatasetInit:
    """Unit tests for 'declearn.dataset.InMemoryDataset' instantiation."""

    def test_from_inputs(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating with (x, y, w) array data."""
        # Split data into distinct objects with various types.
        y_dat = dataset.pop("col_y")
        w_dat = dataset.pop("col_w").values
        x_dat = scipy.sparse.coo_matrix(dataset.values)
        # Test that an InMemoryDataset can be instantiated from that data.
        dst = InMemoryDataset(data=x_dat, target=y_dat, s_wght=w_dat)
        assert dst.feats is x_dat
        assert dst.target is y_dat
        assert dst.weights is w_dat

    def test_from_dataframe(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating with a pandas DataFrame and column names."""
        dst = InMemoryDataset(data=dataset, target="col_y", s_wght="col_w")
        assert np.allclose(dst.feats, dataset[["col_a", "col_b"]])
        assert np.allclose(dst.target, dataset["col_y"])  # type: ignore
        assert np.allclose(dst.weights, dataset["col_w"])  # type: ignore

    def test_from_dataframe_with_fcols_str(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating from a pandas Dataframe with string f_cols."""
        dst = InMemoryDataset(
            data=dataset, target="col_y", s_wght="col_w", f_cols=["col_a"]
        )
        assert np.allclose(dst.feats, dataset[["col_a"]])
        assert np.allclose(dst.target, dataset["col_y"])  # type: ignore
        assert np.allclose(dst.weights, dataset["col_w"])  # type: ignore

    def test_from_dataframe_with_fcols_int(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating from a pandas Dataframe with string f_cols."""
        dst = InMemoryDataset(
            data=dataset, target="col_y", s_wght="col_w", f_cols=[1]
        )
        assert np.allclose(dst.feats, dataset[["col_b"]])
        assert np.allclose(dst.target, dataset["col_y"])  # type: ignore
        assert np.allclose(dst.weights, dataset["col_w"])  # type: ignore

    def test_from_csv_file(
        self,
        dataset: pd.DataFrame,
        tmp_path: str,
    ) -> None:
        """Test instantiating from a single csv file and column names."""
        # Dump the dataset to a csv file and instantiate from it.
        path = os.path.join(tmp_path, "dataset.csv")
        dataset.to_csv(path, index=False)
        dst = InMemoryDataset(data=path, target="col_y", s_wght="col_w")
        # Test that the data matches expectations.
        assert np.allclose(dst.feats, dataset[["col_a", "col_b"]])
        assert np.allclose(dst.target, dataset["col_y"])  # type: ignore
        assert np.allclose(dst.weights, dataset["col_w"])  # type: ignore

    def test_from_csv_file_feats_only(
        self,
        dataset: pd.DataFrame,
        tmp_path: str,
    ) -> None:
        """Test instantiating from a single csv file without y nor w."""
        # Dump the dataset to a csv file and instantiate from it.
        path = os.path.join(tmp_path, "dataset.csv")
        dataset.to_csv(path, index=False)
        dst = InMemoryDataset(data=path)
        # Test that the data matches expectations.
        assert np.allclose(dst.feats, dataset)
        assert dst.target is None
        assert dst.weights is None

    def test_from_data_files(
        self,
        dataset: pd.DataFrame,
        tmp_path: str,
    ) -> None:
        """Test instantiating from a collection of files."""
        # Split data into distinct objects with various types.
        y_dat = dataset.pop("col_y")
        w_dat = dataset.pop("col_w").values
        x_dat = scipy.sparse.coo_matrix(dataset.values)
        # Save these objects to files.
        x_path = save_data_array(os.path.join(tmp_path, "data_x"), x_dat)
        y_path = save_data_array(os.path.join(tmp_path, "data_y"), y_dat)
        w_path = save_data_array(os.path.join(tmp_path, "data_w"), w_dat)
        # Tes that an InMemoryDataset can be instantiated from these files.
        dst = InMemoryDataset(data=x_path, target=y_path, s_wght=w_path)
        assert isinstance(dst.feats, scipy.sparse.coo_matrix)
        assert np.allclose(dst.feats.toarray(), x_dat.toarray())
        assert isinstance(dst.target, pd.Series)
        assert np.allclose(dst.target, y_dat)
        assert isinstance(dst.weights, np.ndarray)
        assert np.allclose(dst.weights, w_dat)  # type: ignore

    def test_from_svmlight(
        self,
        dataset: pd.DataFrame,
        tmp_path: str,
    ) -> None:
        """Test instantiating from a svmlight file."""
        path = os.path.join(tmp_path, "dataset.svmlight")
        sklearn.datasets.dump_svmlight_file(
            scipy.sparse.coo_matrix(dataset[["col_a", "col_b"]].values),
            dataset["col_y"].values,
            path,
        )
        dst = InMemoryDataset.from_svmlight(path)
        assert isinstance(dst.data, scipy.sparse.csr_matrix)
        assert np.allclose(
            dst.data.toarray(), dataset[["col_a", "col_b"]].values
        )
        assert isinstance(dst.target, np.ndarray)
        assert np.allclose(dst.target, dataset["col_y"].to_numpy())
        assert dst.weights is None


class TestInMemoryDatasetProperties:
    """Unit tests for 'declearn.dataset.InMemoryDataset' properties."""

    def test_classes_array(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test (authorized) classes access with numpy array targets."""
        dst = InMemoryDataset(
            data=dataset, target=dataset["col_y"].values, expose_classes=True
        )
        assert dst.classes == {0, 1, 2}

    def test_classes_series(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test (authorized) classes access with pandas Series targets."""
        dst = InMemoryDataset(
            data=dataset, target="col_y", expose_classes=True
        )
        assert dst.classes == {0, 1, 2}

    def test_classes_dataframe(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test (authorized) classes access with pandas DataFrame targets."""
        dst = InMemoryDataset(
            data=dataset, target=dataset[["col_y"]], expose_classes=True
        )
        assert dst.classes == {0, 1, 2}

    def test_classes_sparse(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test (authorized) classes access with scipy spmatrix targets."""
        y_dat = scipy.sparse.coo_matrix(dataset[["col_y"]] + 1)
        dst = InMemoryDataset(data=dataset, target=y_dat, expose_classes=True)
        assert dst.classes == {1, 2, 3}

    def test_data_type_dataframe(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test (authorized) data-type access with pandas DataFrame data."""
        dst = InMemoryDataset(
            data=dataset[["col_a", "col_b"]], expose_data_type=True
        )
        assert dst.data_type == "float32"

    def test_data_type_dataframe_mixed(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test that an exception is raised with a mixed-type DataFrame."""
        dst = InMemoryDataset(data=dataset, expose_data_type=True)
        with pytest.raises(ValueError):
            dst.data_type  # pylint: disable=pointless-statement  # noqa: B018

    def test_data_type_series(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test (authorized) data-type access with pandas Series data."""
        dst = InMemoryDataset(data=dataset["col_a"], expose_data_type=True)
        assert dst.data_type == "float32"

    def test_data_type_array(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test (authorized) data-type access with numpy array data."""
        data = dataset[["col_a", "col_b"]].values
        dst = InMemoryDataset(data=data, expose_data_type=True)
        assert dst.data_type == "float32"

    def test_data_type_sparse(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test (authorized) data-type access with scipy spmatrix data."""
        data = scipy.sparse.coo_matrix(dataset[["col_a", "col_b"]].values)
        dst = InMemoryDataset(data=data, expose_data_type=True)
        assert dst.data_type == "float32"


class TestInMemoryDatasetSaveLoad:
    """Test JSON-file saving/loading features of InMemoryDataset."""

    def test_save_load_json(
        self,
        dataset: pd.DataFrame,
        tmp_path: str,
    ) -> None:
        """Test that a dataset can be saved to and loaded from JSON."""
        dst = InMemoryDataset(dataset, target="col_y", s_wght="col_w")
        # Test that the dataset can be saved to JSON.
        path = os.path.join(tmp_path, "dataset.json")
        dst.save_to_json(path)
        assert os.path.isfile(path)
        # Test that it can be reloaded from JSON.
        bis = InMemoryDataset.load_from_json(path)
        assert np.allclose(dst.data, bis.data)
        assert np.allclose(dst.target, bis.target)  # type: ignore
        assert np.allclose(dst.weights, bis.weights)  # type: ignore
        assert dst.f_cols == bis.f_cols
        assert dst.expose_classes == bis.expose_classes
        assert dst.expose_data_type == bis.expose_data_type

    def test_load_json_malformed(
        self,
        tmp_path: str,
    ) -> None:
        """Test with a JSON file that has nothing to do with a dataset."""
        path = os.path.join(tmp_path, "dataset.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"not-a-dataset": "at-all"}, file)
        with pytest.raises(KeyError):
            InMemoryDataset.load_from_json(path)

    def test_load_json_partial(
        self,
        tmp_path: str,
    ) -> None:
        """Test with a JSON file that contains a partial dataset config."""
        path = os.path.join(tmp_path, "dataset.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"config": {"data": "mock", "target": "mock"}}, file)
        with pytest.raises(KeyError):
            InMemoryDataset.load_from_json(path)
