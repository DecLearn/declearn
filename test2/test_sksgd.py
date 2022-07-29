# coding: utf-8

"""Unit tests for SklearnSGDModel."""

import json
from typing import Iterator, Optional

import numpy as np
import pytest
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.linear_model import SGDClassifier, SGDRegressor  # type: ignore

from declearn2.model.api import NumpyVector
from declearn2.model.sklearn import SklearnSGDModel
from declearn2.typing import Batch


def build_dataset(
        n_classes: Optional[int],
        s_weights: bool,
        as_sparse: bool,
    ) -> Iterator[Batch]:
    """Return a random-valued dataset for a classif. or regress. task."""
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
        yield from zip(inputs, labels, s_wght)
    else:
        yield from zip(inputs, labels, [None, None])


@pytest.mark.parametrize("as_sparse", [False, True], ids=["", "Sparse"])
@pytest.mark.parametrize("s_weights", [False, True], ids=["", "SmpWgt"])
@pytest.mark.parametrize("n_classes", [None, 2, 5], ids=["Reg", "Bin", "Clf"])
def test_model(
        n_classes: Optional[int],
        s_weights: bool,
        as_sparse: bool,
    ) -> None:
    """Unit-test functionalities of a SklearnSGDModel.

    n_classes: int or None
        Specify the learning task: regression if None,
        else binary or multiclass classification.
    s_weights: bool
        Specify whether to use sample weights.
    as_sparse: bool
        Specify whether to return input features as a sparse
        matrix object (note: it will not actually be sparse).
    """
    # Instantiate the model and generate testing data.
    skmod = (SGDClassifier if n_classes else SGDRegressor)()
    model = SklearnSGDModel(skmod, n_features=8, n_classes=n_classes)
    batches = list(build_dataset(n_classes, s_weights, as_sparse))
    # Check that the model can be JSON-(de)serialized properly.
    config = json.dumps(model.get_config())
    other = model.from_config(json.loads(config))
    assert model.get_config() == other.get_config()
    # Check that weights are properly initialized to zero.
    w_srt = model.get_weights()
    assert isinstance(w_srt, NumpyVector)
    assert set(w_srt.coefs.keys()) == {'intercept', 'coef'}
    assert all(np.all(arr == 0.) for arr in w_srt.coefs.values())
    # Check that gradients computation works and leaves weights unaltered.
    grads = model.compute_batch_gradients(batches[0])
    w_end = model.get_weights()
    assert w_srt == w_end
    assert isinstance(grads, NumpyVector)
    assert grads != w_srt  # i.e. comparable but not all-zeros
    # Test that updates' application is mathematically correct.
    grads = -.1 * grads  # type: ignore
    assert isinstance(grads, NumpyVector)
    model.apply_updates(grads)
    w_end = model.get_weights()
    assert w_end != w_srt
    assert w_end == (w_srt + grads)
    # Test that loss computation abides by its specs.
    loss = model.compute_loss(batches)
    assert isinstance(loss, float)
