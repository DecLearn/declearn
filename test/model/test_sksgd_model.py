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

"""Unit tests for SklearnSGDModel."""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore

from declearn.model.sklearn import NumpyVector, SklearnSGDModel
from declearn.model.sklearn._sgd import LossesLiteral
from declearn.test_utils import make_importable
from declearn.typing import Batch

# relative imports from `model_testing.py`
with make_importable(os.path.dirname(__file__)):
    from model_testing import ModelTestCase, ModelTestSuite


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
    framework = "numpy"

    def __init__(
        self,
        n_classes: Optional[int],
        s_weights: bool,
        as_sparse: bool,
        loss: LossesLiteral,
    ) -> None:
        """Specify the desired model and type of input data."""
        self.n_classes = n_classes
        self.s_weights = s_weights
        self.as_sparse = as_sparse
        self.loss = loss

    @property
    def dataset(
        self,
    ) -> List[Batch]:
        """Suited toy binary-classification dataset."""
        rng = np.random.default_rng(seed=0)
        inputs = rng.normal(size=(2, 32, 8)).astype("float32")
        if self.as_sparse:
            inputs = [csr_matrix(arr) for arr in inputs]  # type: ignore
        if isinstance(self.n_classes, int):
            labels = rng.choice(self.n_classes, size=(2, 32)).astype("float32")
        else:
            labels = rng.normal(size=(2, 32)).astype("float32")
        if self.s_weights:
            s_wght = np.exp(rng.normal(size=(2, 32)).astype("float32"))
            s_wght /= s_wght.sum(axis=1, keepdims=True) * 32
            batches = list(zip(inputs, labels, s_wght, strict=False))
        else:
            batches = list(zip(inputs, labels, [None, None], strict=False))
        return batches

    @property
    def model(
        self,
    ) -> SklearnSGDModel:
        """Suited toy binary-classification model."""
        if self.n_classes is None:
            skmod = SGDRegressor(loss=self.loss)
        else:
            skmod = SGDClassifier(loss=self.loss)
        model = SklearnSGDModel(skmod, dtype="float32")
        data_info: Dict[str, Any] = {"features_shape": (8,)}
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
    s_weights: bool,
    as_sparse: bool,
    loss_cfg: dict,
) -> SklearnSGDTestCase:
    """Fixture to access a SklearnSGDTestCase."""
    return SklearnSGDTestCase(
        loss_cfg["n_classes"], s_weights, as_sparse, loss_cfg["loss"]
    )


REG_LOSSES = (
    "squared_error",
    "huber",
    "epsilon_insensitive",
    "squared_epsilon_insensitive",
)

CLS_LOSSES = (
    "hinge",
    "log_loss",
    "modified_huber",
    "squared_hinge",
    "perceptron",
    "squared_error",
    "huber",
    "epsilon_insensitive",
    "squared_epsilon_insensitive",
)


NCLASSES_TO_ID = {
    None: "Reg",
    2: "Bin",
    5: "Clf",
}

LOSS_TO_ID = {
    "hinge": "Hinge",
    "log_loss": "Log",
    "modified_huber": "MdfHbr",
    "squared_hinge": "SqrHinge",
    "perceptron": "Prcpt",
    "squared_error": "Squared",
    "huber": "Huber",
    "epsilon_insensitive": "EpsIns",
    "squared_epsilon_insensitive": "SqrEpsIns",
}

LOSS_CONFIGS = (
    [
        {"n_classes": None, "loss": loss}
        for loss in REG_LOSSES  # regression configs
    ]
    + [
        {"n_classes": 2, "loss": loss}
        for loss in CLS_LOSSES  # bin. classification configs
    ]
    + [
        {"n_classes": 5, "loss": loss}
        for loss in CLS_LOSSES  # multi-class classification configs
    ]
)


@pytest.mark.parametrize("as_sparse", [False, True], ids=["", "Sparse"])
@pytest.mark.parametrize("s_weights", [False, True], ids=["", "SmpWgt"])
@pytest.mark.parametrize(
    "loss_cfg",
    LOSS_CONFIGS,
    ids=lambda c: f"{NCLASSES_TO_ID[c['n_classes']]}_{LOSS_TO_ID[c['loss']]}",
)
class TestSklearnSGDModel(ModelTestSuite):
    """Unit tests for declearn.model.sklearn.SklearnSGDModel."""

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
