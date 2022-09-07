# coding: utf-8

"""Shared testing code for TensorFlow and Torch models' unit tests."""

import json
from typing import Any, List, Protocol, Tuple, Type, Union

import numpy as np

from declearn2.model.api import Model, NumpyVector, Vector
from declearn2.typing import Batch
from declearn2.utils import json_pack, json_unpack


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
        assert isinstance(w_srt, NumpyVector)
        w_end = w_srt + 1.
        model.set_weights(w_end)  # type: ignore
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
        np_batch = [
            None if arr is None else test_case.to_numpy(arr)
            for arr in my_batch
        ]
        assert isinstance(np_batch[0], np.ndarray)
        # Compute gradients in both cases.
        np_grads = model.compute_batch_gradients(np_batch)  # type: ignore
        assert isinstance(np_grads, test_case.vector_cls)
        my_grads = model.compute_batch_gradients(my_batch)
        assert my_grads == np_grads

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
        grads = -.1 * grads
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
        loss = test_case.model.compute_loss(test_case.dataset)
        assert isinstance(loss, float)
