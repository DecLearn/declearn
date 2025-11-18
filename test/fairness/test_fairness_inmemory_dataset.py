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

"""Unit tests for 'declearn.fairness.core.FairnessInMemoryDataset'"""

import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix  # type: ignore

from declearn.dataset import InMemoryDataset
from declearn.fairness.core import FairnessInMemoryDataset

SEED = 0


@pytest.fixture(name="dataset")
def dataset_fixture() -> pd.DataFrame:
    """Fixture providing with a small toy dataset."""
    rng = np.random.default_rng(seed=SEED)
    wgt = rng.normal(size=100).astype("float32")
    data = {
        "col_a": np.arange(100, dtype="float32"),
        "col_b": rng.normal(size=100).astype("float32"),
        "col_y": rng.choice(2, size=100, replace=True),
        "col_w": wgt / sum(wgt),
        "col_s": rng.choice(2, size=100, replace=True),
    }
    return pd.DataFrame(data)


class TestFairnessInMemoryDatasetInit:
    """Unit tests for 'declearn.fairness.core.FairnessInMemoryDataset' init."""

    def test_init_sattr_dataframe_target_none(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating with a sensitive attribute and no target."""
        s_attr = pd.DataFrame(dataset.pop("col_s"))
        with pytest.warns(RuntimeWarning):  # due to sensitive_target=True
            dst = FairnessInMemoryDataset(
                dataset, s_attr=s_attr, sensitive_target=True
            )
        assert isinstance(dst.sensitive, pd.Series)
        assert (dst.sensitive == s_attr.apply(tuple, axis=1)).all()

    def test_init_sattr_dataframe_target(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating with both sensitive attribute and target."""
        s_attr = pd.DataFrame(dataset.pop("col_s"))
        dst = FairnessInMemoryDataset(
            dataset, s_attr=s_attr, target="col_y", sensitive_target=True
        )
        expected = pd.DataFrame(
            {"target": dataset["col_y"], "col_s": s_attr["col_s"]}
        ).apply(tuple, axis=1)
        assert isinstance(dst.sensitive, pd.Series)
        assert (dst.sensitive == expected).all()

    def test_init_sattr_dataframe_no_target(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating with a sensitive attribute, ignoring target."""
        s_attr = pd.DataFrame(dataset.pop("col_s"))
        dst = FairnessInMemoryDataset(
            dataset, s_attr=s_attr, target="col_y", sensitive_target=False
        )
        assert isinstance(dst.sensitive, pd.Series)
        assert (dst.sensitive == s_attr.apply(tuple, axis=1)).all()

    def test_init_sattr_one_column(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating with sensitive attributes as a column name."""
        dst = FairnessInMemoryDataset(
            dataset, s_attr=["col_s"], sensitive_target=False
        )
        expected = dataset[["col_s"]].apply(tuple, axis=1)
        assert (dst.sensitive == expected).all()

    def test_init_sattr_multiple_columns(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating with sensitive attributes as column names."""
        dst = FairnessInMemoryDataset(
            dataset, s_attr=["col_s", "col_y"], sensitive_target=False
        )
        expected = dataset[["col_s", "col_y"]].apply(tuple, axis=1)
        assert (dst.sensitive == expected).all()

    def test_init_sattr_column_indices(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating with sensitive attributes as column indices."""
        dst = FairnessInMemoryDataset(
            dataset, s_attr=[2, 4], sensitive_target=False
        )
        expected = dataset.iloc[:, [2, 4]].apply(tuple, axis=1)
        assert (dst.sensitive == expected).all()

    def test_init_sattr_path(
        self,
        dataset: pd.DataFrame,
        tmp_path: str,
    ) -> None:
        """Test instantiating with a sensitive attribute as a file dump."""
        path = os.path.join(tmp_path, "s_attr.csv")
        s_attr = dataset.pop("col_s")
        s_attr.to_csv(path, index=False)
        dst = FairnessInMemoryDataset(
            dataset, s_attr=path, target="col_y", sensitive_target=True
        )
        expected = pd.DataFrame(
            {"target": dataset["col_y"], "col_s": s_attr}
        ).apply(tuple, axis=1)
        assert (dst.sensitive == expected).all()

    def test_init_sattr_array(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating with a sensitive attribute array."""
        s_attr = dataset.pop("col_s").values
        dst = FairnessInMemoryDataset(
            dataset, s_attr=s_attr, sensitive_target=False
        )
        assert isinstance(dst.sensitive, pd.Series)
        expected = pd.DataFrame(s_attr).apply(tuple, axis=1)
        assert (dst.sensitive == expected).all()

    def test_init_sattr_spmatrix(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test instantiating with a sensitive attribute spmatrix."""
        s_attr = coo_matrix(dataset.pop("col_s").values).T
        dst = FairnessInMemoryDataset(
            dataset, s_attr=s_attr, sensitive_target=False
        )
        assert isinstance(dst.sensitive, pd.Series)
        expected = pd.DataFrame(s_attr.toarray()).apply(tuple, axis=1)
        assert (dst.sensitive == expected).all()

    def test_init_error_wrong_sattr_cols(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test that an error is raised if 's_attr' is a wrongful list."""
        with pytest.raises(TypeError):
            FairnessInMemoryDataset(
                dataset, s_attr=["col_wrong"], sensitive_target=False
            )

    def test_init_error_wrong_sattr_type(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test that an error is raised if 's_attr' has unsupported type."""
        with pytest.raises(TypeError):
            FairnessInMemoryDataset(
                dataset, s_attr=mock.MagicMock(), sensitive_target=False
            )

    def test_init_error_wrong_sattr_shape(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """Test that an error is raised if 's_attr' has unsupported type."""
        s_attr = dataset.pop("col_s").values[:20]
        with pytest.raises(ValueError):
            FairnessInMemoryDataset(
                dataset, s_attr=s_attr, sensitive_target=False
            )


@pytest.fixture(name="fdst")
def fdst_fixture(
    dataset: pd.DataFrame,
) -> FairnessInMemoryDataset:
    """Fixture providing with a wrapped small toy dataset."""
    return FairnessInMemoryDataset(
        data=pd.DataFrame(dataset),
        target="col_y",
        s_wght="col_w",
        s_attr=["col_s"],
        sensitive_target=True,
    )


class TestFairnessInMemoryDataset:
    """Unit tests for 'declearn.fairness.core.FairnessInMemoryDataset'."""

    def test_get_sensitive_group_definitions(
        self,
        fdst: FairnessInMemoryDataset,
    ) -> None:
        """Test that sensitive groups definitions match expectations."""
        groups = fdst.get_sensitive_group_definitions()
        assert groups == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_get_sensitive_group_counts(
        self,
        fdst: FairnessInMemoryDataset,
    ) -> None:
        """Test that sensitive group counts match wrapped data."""
        counts = fdst.get_sensitive_group_counts()
        assert set(counts) == set(fdst.get_sensitive_group_definitions())
        assert isinstance(fdst.data, pd.DataFrame)  # by construction here
        assert counts == fdst.data[["col_y", "col_s"]].value_counts().to_dict()

    def test_get_sensitive_group_subset(
        self,
        fdst: FairnessInMemoryDataset,
    ) -> None:
        """Test that sensitive group subset access works properly."""
        subset = fdst.get_sensitive_group_subset(group=(0, 0))
        assert isinstance(subset, InMemoryDataset)
        assert (subset.target == 0).all()
        assert (subset.feats["col_s"] == 0).all()
        assert len(subset.data) == fdst.get_sensitive_group_counts()[(0, 0)]

    def test_set_sensitive_group_weights(
        self,
        fdst: FairnessInMemoryDataset,
    ) -> None:
        """Test that sensitive group weighting works properly."""
        # Assert that initial sample weights are based on specified column.
        assert isinstance(fdst.data, pd.DataFrame)
        expected = fdst.data["col_w"]
        assert (fdst.weights == expected).all()
        # Set sensitive group weights.
        weights = {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.3, (1, 1): 0.4}
        fdst.set_sensitive_group_weights(weights, adjust_by_counts=False)
        # Assert that resuling sample weights match expectations.
        sgroup = fdst.data[["col_y", "col_s"]].apply(tuple, axis=1)
        expected = fdst.data["col_w"] * sgroup.apply(weights.get)
        assert (fdst.weights == expected).all()
        # Do it again with counts-adjustment.
        counts = fdst.get_sensitive_group_counts()
        weights = {key: val / counts[key] for key, val in weights.items()}
        fdst.set_sensitive_group_weights(weights, adjust_by_counts=True)
        assert np.allclose(fdst.weights, expected)

    def test_set_sensitive_group_weights_keyerror(
        self,
        fdst: FairnessInMemoryDataset,
    ) -> None:
        """Test setting sensitive group weights with missing groups."""
        weights = {(0, 0): 0.1, (0, 1): 0.2}
        with pytest.raises(KeyError):
            fdst.set_sensitive_group_weights(weights)
