# coding: utf-8

"""Shared testing code for TensorFlow and Torch models' unit tests."""

import json
from typing import Any, List, Protocol, Tuple, Type, Union

import numpy as np
import pytest

from declearn.model.api import Model, Vector
from declearn.typing import Batch
from declearn.utils import json_pack, json_unpack


class ModelTestCase(Protocol):
    """TestCase fixture-provider protocol."""

    vector_cls: Type[Vector]
    tensor_cls: Union[Type[Any], Tuple[Type[Any], ...]]

    @staticmethod
    def to_numpy(
        tensor: Any,
    ) -> np.ndarray:
        """Convert an input tensor to a numpy array."""

    @property
    def dataset(
        self,
    ) -> List[Batch]:
        """Suited toy binary-classification dataset."""

    @property
    def model(
        self,
    ) -> Model:
        """Suited toy binary-classification model."""


class ModelTestSuite:
    """Unit tests for a declearn Model."""

    def test_serialization(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that the model can be JSON-(de)serialized properly."""
        model = test_case.model
        config = json.dumps(model.get_config())
        other = model.from_config(json.loads(config))
        assert model.get_config() == other.get_config()

    def test_get_set_weights(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that weights can properly be accessed and replaced."""
        model = test_case.model
        w_srt = model.get_weights()
        assert isinstance(w_srt, test_case.vector_cls)
        w_end = w_srt + 1.0
        model.set_weights(w_end)
        assert model.get_weights() == w_end

    def test_compute_batch_gradients(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that gradients computation works."""
        # Setup the model and a batch of data.
        model = test_case.model
        batch = test_case.dataset[0]
        # Check that gradients computation works.
        w_srt = model.get_weights()
        grads = model.compute_batch_gradients(batch)
        w_end = model.get_weights()
        assert w_srt == w_end
        assert isinstance(grads, test_case.vector_cls)

    def test_compute_batch_gradients_np(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that gradients computations work with numpy inputs."""
        # Setup the model and a batch of data, in both tf and np formats.
        model = test_case.model
        my_batch = test_case.dataset[0]
        assert isinstance(my_batch[0], test_case.tensor_cls)
        np_batch = tuple(
            None if arr is None else test_case.to_numpy(arr)
            for arr in my_batch
        )
        assert isinstance(np_batch[0], np.ndarray)
        # Compute gradients in both cases.
        np_grads = model.compute_batch_gradients(np_batch)  # type: ignore
        assert isinstance(np_grads, test_case.vector_cls)
        my_grads = model.compute_batch_gradients(my_batch)
        assert my_grads == np_grads

    def test_compute_batch_gradients_clipped(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Check that gradients computation with sample-wise clipping works."""
        # NOTE: this test does not check that results are correct
        # Setup the model and a batch of data.
        model = test_case.model
        batch = test_case.dataset[0]
        # Check that gradients computation works.
        w_srt = model.get_weights()
        grads_a = model.compute_batch_gradients(batch, max_norm=None)
        grads_b = model.compute_batch_gradients(batch, max_norm=0.05)
        w_end = model.get_weights()
        assert w_srt == w_end
        assert isinstance(grads_b, test_case.vector_cls)
        assert grads_a.coefs.keys() == grads_b.coefs.keys()
        assert all(
            grads_a.coefs[k].shape == grads_b.coefs[k].shape
            for k in grads_a.coefs
        )
        assert grads_a != grads_b

    def test_apply_updates(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Test that updates' application is mathematically correct."""
        model = test_case.model
        batch = test_case.dataset[0]
        # Compute gradients.
        w_srt = model.get_weights()
        grads = model.compute_batch_gradients(batch)
        # Check that updates can be obtained and applied.
        grads = -0.1 * grads
        assert isinstance(grads, test_case.vector_cls)
        model.apply_updates(grads)
        # Verify the the updates were correctly applied.
        # Check up to 1e-6 numerical precision due to tensor/np conversion.
        # NOTE: if the model had frozen weights, this test would xfail.
        w_end = model.get_weights()
        assert w_end != w_srt
        updt = [test_case.to_numpy(val) for val in grads.coefs.values()]
        diff = list((w_end - w_srt).coefs.values())
        assert all(np.abs(a - b).max() < 1e-6 for a, b in zip(diff, updt))

    def test_serialize_gradients(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Test that computed gradients can be (de)serialized as strings."""
        model = test_case.model
        batch = test_case.dataset[0]
        grads = model.compute_batch_gradients(batch)
        gdump = json.dumps(grads, default=json_pack)
        assert isinstance(gdump, str)
        other = json.loads(gdump, object_hook=json_unpack)
        assert grads == other

    def test_compute_loss(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Test that loss computation abides by its specs."""
        with pytest.warns(DeprecationWarning):
            loss = test_case.model.compute_loss(test_case.dataset)
        assert isinstance(loss, float)

    def test_compute_batch_predictions(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Test that predictions' computation abids by its specs."""
        model = test_case.model
        batch = test_case.dataset[0]
        y_true, y_pred, s_wght = model.compute_batch_predictions(batch)
        assert isinstance(y_pred, np.ndarray) and y_pred.ndim >= 1
        assert isinstance(y_true, np.ndarray) and y_true.ndim >= 1
        assert len(y_pred) == len(y_true)
        if batch[2] is None:
            assert s_wght is None
        else:
            assert isinstance(s_wght, np.ndarray) and s_wght.ndim == 1
            assert len(s_wght) == len(y_true)

    def test_loss_function(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Test that the exposed loss function abides by its specs."""
        model = test_case.model
        batch = test_case.dataset[0]
        y_true, y_pred, s_wght = model.compute_batch_predictions(batch)
        s_loss = model.loss_function(y_true, y_pred).squeeze()
        assert isinstance(s_loss, np.ndarray) and s_loss.ndim == 1
        assert len(s_loss) == len(s_wght if s_wght is not None else y_true)
        if s_wght is None:
            s_wght = np.ones_like(s_loss)
        r_loss = (s_loss * s_wght).sum() / s_wght.sum()
        with pytest.warns(DeprecationWarning):
            loss = model.compute_loss([batch])
        assert r_loss == loss
