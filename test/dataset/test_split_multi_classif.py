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

"""Unit tests for 'declearn.dataset.utils.split_multi_classif_dataset'."""

from typing import List, Tuple, Type, Union

import numpy as np
import pytest
from scipy.sparse import coo_matrix, spmatrix  # type: ignore
from scipy.stats import chi2_contingency  # type: ignore

from declearn.dataset.utils import split_multi_classif_dataset

Array = Union[np.ndarray, spmatrix]


@pytest.fixture(name="dataset", scope="module")
def dataset_fixture(
    sparse: bool,
) -> Tuple[Array, np.ndarray]:
    """Fixture providing with a random classification dataset."""
    rng = np.random.default_rng(seed=0)
    x_dat = rng.normal(size=(400, 32))
    y_dat = rng.choice(4, size=400)
    if sparse:
        return coo_matrix(x_dat), y_dat
    return x_dat, y_dat


@pytest.mark.parametrize(
    "sparse", [False, True], ids=["dense", "sparse"], scope="module"
)
class TestSplitMultiClassifDataset:
    """Unit tests for `split_multi_classif_dataset`."""

    @classmethod
    def assert_expected_shard_shapes(
        cls,
        dataset: Tuple[Array, np.ndarray],
        shards: List[
            Tuple[Tuple[Array, np.ndarray], Tuple[Array, np.ndarray]]
        ],
        n_shards: int,
        p_valid: float,
    ) -> None:
        """Verify that shards match expected shapes."""
        assert isinstance(shards, list) and len(shards) == n_shards
        n_samples = 0
        for shard in shards:
            cls._assert_valid_shard(
                shard,
                n_feats=dataset[0].shape[1],
                n_label=len(np.unique(dataset[1])),
                p_valid=p_valid,
                x_type=type(dataset[0]),
            )
            n_samples += shard[0][0].shape[0] + shard[1][0].shape[0]
        assert n_samples == dataset[0].shape[0]

    @staticmethod
    def _assert_valid_shard(
        shard: Tuple[Tuple[Array, np.ndarray], Tuple[Array, np.ndarray]],
        n_feats: int = 32,
        n_label: int = 4,
        p_valid: float = 0.2,
        x_type: Union[Type[np.ndarray], Type[spmatrix]] = np.ndarray,
    ) -> None:
        """Assert that a given dataset shard matches expected specs."""
        # Unpack arrays and verify that their types are coherent.
        (x_train, y_train), (x_valid, y_valid) = shard
        assert isinstance(x_train, x_type)
        assert isinstance(x_valid, x_type)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_valid, np.ndarray)
        # Assert that array shapes match expectations.
        assert x_train.ndim == x_valid.ndim == 2  # type: ignore
        assert x_train.shape[0] == y_train.shape[0]
        assert x_valid.shape[0] == y_valid.shape[0]
        assert x_train.shape[1] == x_valid.shape[1] == n_feats
        assert y_train.ndim == y_valid.ndim == 1
        # Assert that labels have proper values.
        labels = list(range(n_label))
        assert np.all(np.isin(y_train, labels))
        assert np.all(np.isin(y_valid, labels))
        # Assert that train/valid partition matches expectation.
        s_valid = x_valid.shape[0] / (x_train.shape[0] + x_valid.shape[0])
        assert abs(p_valid - s_valid) <= 0.02

    @staticmethod
    def get_label_counts(
        shards: List[
            Tuple[Tuple[Array, np.ndarray], Tuple[Array, np.ndarray]]
        ],
        n_label: int = 4,
    ) -> np.ndarray:
        """Return the shard-wise subset-wise label counts, stacked together."""
        counts = [
            np.bincount(y, minlength=n_label)
            for (_, y_train), (_, y_valid) in shards
            for y in (y_train, y_valid)
        ]
        return np.stack(counts)

    def test_scheme_iid(
        self,
        dataset: Tuple[Array, np.ndarray],
    ) -> None:
        """Test that the iid scheme yields iid samples."""
        shards = split_multi_classif_dataset(
            dataset, n_shards=4, scheme="iid", p_valid=0.2, seed=0
        )
        # Verify that shards match expected shapes.
        self.assert_expected_shard_shapes(
            dataset, shards, n_shards=4, p_valid=0.2
        )
        # Verify that labels are iid-distributed across shards.
        # To do so, use a chi2-test with blatantly high acceptance rate
        # for the null hypothesis that distributions differ, and verify
        # that the hypothesis would still be rejected at that rate.
        y_counts = self.get_label_counts(shards)
        assert chi2_contingency(y_counts).pvalue >= 0.90

    def test_scheme_labels(
        self,
        dataset: Tuple[Array, np.ndarray],
    ) -> None:
        """Test that the labels scheme yields non-overlapping-labels shards."""
        shards = split_multi_classif_dataset(
            dataset, n_shards=2, scheme="labels", p_valid=0.4, seed=0
        )
        # Verify that shards match expected shapes.
        self.assert_expected_shard_shapes(
            dataset, shards, n_shards=2, p_valid=0.4
        )
        # Verify that labels are distributed without overlap across shards.
        labels_train_0 = np.unique(shards[0][0][1])
        labels_valid_0 = np.unique(shards[0][1][1])
        labels_train_1 = np.unique(shards[1][0][1])
        labels_valid_1 = np.unique(shards[1][1][1])
        assert np.all(labels_train_0 == labels_valid_0)
        assert np.all(labels_train_1 == labels_valid_1)
        assert np.intersect1d(labels_train_0, labels_train_1).shape == (0,)

    def test_scheme_biased(
        self,
        dataset: Tuple[Array, np.ndarray],
    ) -> None:
        """Test that the biased scheme yields disparate-distrib. shards."""
        shards = split_multi_classif_dataset(
            dataset, n_shards=4, scheme="biased", p_valid=0.2, seed=0
        )
        # Verify that shards match expected shapes.
        self.assert_expected_shard_shapes(
            dataset, shards, n_shards=4, p_valid=0.2
        )
        # Verify that labels have distinct distributions across shards.
        # To do so, use a chi2-test (with the null hypothesis that
        # distributions differ), and verify that the hypothesis is
        # accepted overall with high confidence, and rejected on
        # train/valid pairs with high confidence as well.
        y_counts = self.get_label_counts(shards)
        assert chi2_contingency(y_counts).pvalue <= 1e-10
        for i in range(len(shards)):
            assert chi2_contingency(y_counts[i : i + 1]).pvalue >= 0.90

    def test_scheme_dirichlet(
        self,
        dataset: Tuple[Array, np.ndarray],
    ) -> None:
        """Test the dirichlet scheme's disparity, with various alpha values."""
        shards = split_multi_classif_dataset(
            dataset, n_shards=2, scheme="dirichlet", seed=0
        )
        # Verify that shards match expected shapes.
        self.assert_expected_shard_shapes(
            dataset, shards, n_shards=2, p_valid=0.2
        )
        # Verify that labels are differently-distributed across shards.
        # To do so, use a chi2-test (with the null hypothesis that
        # distributions differ), and verify that the hypothesis is
        # accepted overall with high confidence, and rejected on
        # train/valid pairs with high confidence as well.
        y_counts = self.get_label_counts(shards)
        pval_low = chi2_contingency(y_counts).pvalue
        assert pval_low <= 1e-10
        for i in range(len(shards)):
            assert chi2_contingency(y_counts[i : i + 1]).pvalue >= 0.60
        # Verify that using a higher alpha results in a high p-value.
        shards = split_multi_classif_dataset(
            dataset, n_shards=2, scheme="dirichlet", seed=0, alpha=2.0
        )
        y_counts = self.get_label_counts(shards)
        pval_mid = chi2_contingency(y_counts).pvalue
        assert pval_low < pval_mid <= 0.05
        # Verify that using a very high alpha value results in iid data.
        shards = split_multi_classif_dataset(
            dataset, n_shards=2, scheme="dirichlet", seed=0, alpha=100000.0
        )
        y_counts = self.get_label_counts(shards)
        assert chi2_contingency(y_counts).pvalue >= 0.90

    def test_error_labels_too_many_shards(
        self,
        dataset: Tuple[Array, np.ndarray],
    ) -> None:
        """Test that in 'labels' schemes 'n_shards > n_lab' raises an error."""
        with pytest.raises(ValueError):
            split_multi_classif_dataset(dataset, n_shards=8, scheme="labels")
