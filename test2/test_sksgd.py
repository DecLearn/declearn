# coding: utf-8

"""Unit tests for SklearnSGDModel.

Three model/task settings are tested:
* "Reg": Regression (single-target)
* "Bin": Binary classification
* "Clf": Multiclass classification

Random-valued data is generated based on the task.
Additional data settings are tested:
* "-SmpWgt": include sample weights in the yielded batches
* "-Sparse": wrap up the input features as a CSR sparse matrix
"""

import json
from typing import List, Optional

import numpy as np
import pytest
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore

from declearn2.model.api import NumpyVector
from declearn2.model.sklearn import SklearnSGDModel
from declearn2.typing import Batch


@pytest.fixture
def model(
        n_classes: Optional[int],
    ) -> SklearnSGDModel:
    """Instantiate a SklearnSGDModel to test with."""
    skmod = (SGDClassifier if n_classes else SGDRegressor)()
    return SklearnSGDModel(skmod, n_features=8, n_classes=n_classes)


@pytest.fixture
def dataset(
        n_classes: Optional[int],
        s_weights: bool,
        as_sparse: bool,
    ) -> List[Batch]:
    """Return a random-valued dataset for a classif. or regress. task.

    n_classes: int or None
        Specify the learning task: regression if None,
        else binary or multiclass classification.
    s_weights: bool
        Specify whether to use sample weights.
    as_sparse: bool
        Specify whether to return input features as a sparse
        matrix object (note: it will not actually be sparse).
    """
    rng = np.random.default_rng(seed=0)
    inputs = rng.normal(size=(2, 32, 8))
    if as_sparse:
        inputs = [csr_matrix(inputs[0]), csr_matrix(inputs[1])]  # type: ignore
    if isinstance(n_classes, int):
        labels = rng.choice(n_classes, size=(2, 32)).astype(float)
    else:
        labels = rng.normal(size=(2, 32))
    if s_weights:
        s_wght = np.exp(rng.normal(size=(2, 32)))
        s_wght /= s_wght.sum(axis=1, keepdims=True) * 32
        batches = list(zip(inputs, labels, s_wght))
    else:
        batches = list(zip(inputs, labels, [None, None]))
    return batches


@pytest.mark.parametrize("n_classes", [None, 2, 5], ids=["Reg", "Bin", "Clf"])
class TestSklearnSGDModelInit:
    """Unit tests for declearn.model.sklearn.SklearnSGDModel.

    This class groups tests that require only a model object.
    """

    def test_serialization(self, model):
        """Check that the model can be JSON-(de)serialized properly."""
        config = json.dumps(model.get_config())
        other = model.from_config(json.loads(config))
        assert model.get_config() == other.get_config()

    def test_initialization(self, model):
        """Check that weights are properly initialized to zero."""
        w_srt = model.get_weights()
        assert isinstance(w_srt, NumpyVector)
        assert set(w_srt.coefs.keys()) == {'intercept', 'coef'}
        assert all(np.all(arr == 0.) for arr in w_srt.coefs.values())


@pytest.mark.parametrize("as_sparse", [False, True], ids=["", "Sparse"])
@pytest.mark.parametrize("s_weights", [False, True], ids=["", "SmpWgt"])
@pytest.mark.parametrize("n_classes", [None, 2, 5], ids=["Reg", "Bin", "Clf"])
class TestSklearnSGDModelUsage:
    """Unit tests for declearn.model.sklearn.SklearnSGDModel.

    This class groups tests that require both a model and some data.
    """

    def test_compute_batch_gradients(self, model, dataset):
        """Check that gradients computation works."""
        w_srt = model.get_weights()
        grads = model.compute_batch_gradients(dataset[0])
        w_end = model.get_weights()
        assert w_srt == w_end
        assert isinstance(grads, NumpyVector)
        assert not all(np.all(arr == 0.) for arr in grads.coefs.values())
        assert grads.coefs.keys() == w_srt.coefs.keys()

    def test_apply_updates(self, model, dataset):
        """Test that updates' application is mathematically correct."""
        # Compute gradients.
        w_srt = model.get_weights()
        grads = model.compute_batch_gradients(dataset[0])
        # Check that updates can be obtained and applied.
        grads = -.1 * grads
        assert isinstance(grads, NumpyVector)
        model.apply_updates(grads)
        # Verify the the updates were correctly applied.
        w_end = model.get_weights()
        assert w_end != w_srt
        assert w_end == (w_srt + grads)

    def test_compute_loss(self, model, dataset):
        """Test that loss computation abides by its specs."""
        loss = model.compute_loss(dataset)
        assert isinstance(loss, float)
