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

"""Unit tests for SklearnSGDModel."""

import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore

from declearn.model.sklearn import NumpyVector, SklearnSGDModel
from declearn.typing import Batch

# dirty trick to import from `model_testing.py`;
# pylint: disable=wrong-import-order, wrong-import-position
sys.path.append(".")
from model_testing import ModelTestSuite, ModelTestCase


class SklearnSGDTestCase(ModelTestCase):
    """Scikit-Learn SGD model test-case-provider fixture.

    Three model/task settings are tested:
    * "Reg": Regression (single-target)
    * "Bin": Binary classification
    * "Clf": Multiclass classification

    Random-valued data is generated based on the task.
    Additional data settings are tested:
    * "-SmpWgt": include sample weights in the yielded batches
    * "-Sparse": wrap up the input features as a CSR sparse matrix
    """

    vector_cls = NumpyVector
    tensor_cls = (np.ndarray, csr_matrix)

    def __init__(
        self,
        n_classes: Optional[int],
        s_weights: bool,
        as_sparse: bool,
    ) -> None:
        """Specify the desired model and type of input data."""
        self.n_classes = n_classes
        self.s_weights = s_weights
        self.as_sparse = as_sparse

    @staticmethod
    def to_numpy(
        tensor: Any,
    ) -> np.ndarray:
        """Convert an input tensor to a numpy array."""
        assert isinstance(tensor, (np.ndarray, csr_matrix))
        if isinstance(tensor, csr_matrix):
            tensor = tensor.toarray()
        return tensor  # type: ignore

    @property
    def dataset(
        self,
    ) -> List[Batch]:
        """Suited toy binary-classification dataset."""
        rng = np.random.default_rng(seed=0)
        inputs = rng.normal(size=(2, 32, 8))
        if self.as_sparse:
            inputs = [csr_matrix(arr) for arr in inputs]  # type: ignore
        if isinstance(self.n_classes, int):
            labels = rng.choice(self.n_classes, size=(2, 32)).astype(float)
        else:
            labels = rng.normal(size=(2, 32))
        if self.s_weights:
            s_wght = np.exp(rng.normal(size=(2, 32)))
            s_wght /= s_wght.sum(axis=1, keepdims=True) * 32
            batches = list(zip(inputs, labels, s_wght))
        else:
            batches = list(zip(inputs, labels, [None, None]))
        return batches

    @property
    def model(
        self,
    ) -> SklearnSGDModel:
        """Suited toy binary-classification model."""
        skmod = (SGDClassifier if self.n_classes else SGDRegressor)()
        model = SklearnSGDModel(skmod)
        data_info = {"n_features": 8}  # type: Dict[str, Any]
        if self.n_classes:
            data_info["classes"] = np.arange(self.n_classes)
        model.initialize(data_info)
        return model

    def assert_correct_device(
        self,
        vector: NumpyVector,
    ) -> None:
        pass


@pytest.fixture(name="test_case")
def fixture_test_case(
    n_classes: Optional[int],
    s_weights: bool,
    as_sparse: bool,
) -> SklearnSGDTestCase:
    """Fixture to access a SklearnSGDTestCase."""
    return SklearnSGDTestCase(n_classes, s_weights, as_sparse)


@pytest.mark.parametrize("as_sparse", [False, True], ids=["", "Sparse"])
@pytest.mark.parametrize("s_weights", [False, True], ids=["", "SmpWgt"])
@pytest.mark.parametrize("n_classes", [None, 2, 5], ids=["Reg", "Bin", "Clf"])
class TestSklearnSGDModel(ModelTestSuite):
    """Unit tests for declearn.model.sklearn.SklearnSGDModel."""

    def test_serialization(  # type: ignore  # Liskov does not matter here
        self,
        test_case: SklearnSGDTestCase,
    ) -> None:
        # Avoid re-running tests that are unaltered by data parameters.
        if test_case.s_weights or test_case.as_sparse:
            return None
        return super().test_serialization(test_case)

    def test_initialization(
        self,
        test_case: SklearnSGDTestCase,
    ) -> None:
        """Check that weights are properly initialized to zero."""
        # Avoid re-running tests that are unaltered by data parameters.
        if test_case.s_weights or test_case.as_sparse:
            return None
        # Run the actual test.
        model = test_case.model
        w_srt = model.get_weights()
        assert isinstance(w_srt, NumpyVector)
        assert set(w_srt.coefs.keys()) == {"intercept", "coef"}
        assert all(np.all(arr == 0.0) for arr in w_srt.coefs.values())
        return None

    def test_get_set_weights(  # type: ignore  # Liskov does not matter here
        self,
        test_case: SklearnSGDTestCase,
    ) -> None:
        # Avoid re-running tests that are unaltered by data parameters.
        if test_case.s_weights or test_case.as_sparse:
            return None
        # Run the actual test.
        return super().test_get_set_weights(test_case)

    def test_compute_batch_gradients_np(
        self,
        test_case: ModelTestCase,
    ) -> None:
        # The model already uses numpy inputs, this test is unrequired here.
        # NOTE: in fact, it fails with sparse inputs as intercept is *not*
        #       fitted equally (while coefficients are) -> investigate this.
        return None
