"""Shared pytest fixtures for client sampler testing."""

from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pytest

from declearn.aggregator import ModelUpdates
from declearn.messaging import TrainReply

# from declearn.model.api import Model
from declearn.model.api import Model
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
)
from declearn.utils import set_device_policy

# from test.model.model_testing import ModelTestCase


@pytest.fixture(name="client_to_reply")
def client_to_reply_fixture(
    framework: FrameworkType,
    n_clients: int = 3,
) -> Dict[str, TrainReply]:
    """
    Build a dictionary of client replies (messages) from updates that we define
    as vectors with custom value.
    """
    set_device_policy(gpu=False)
    updates_1 = GradientsTestCase(framework).mock_gradient_custom(
        [
            np.array([0, 0, 0, 0]),
            np.array([0, 0]),
        ]
    )
    updates_2 = GradientsTestCase(framework).mock_gradient_custom(
        [
            np.array([1, 1, 1, 1]),
            np.array([1, 1]),
        ]
    )
    updates_3 = GradientsTestCase(framework).mock_gradient_custom(
        [
            np.array([-1, 2, 1, 0]),
            np.array([0.5, -2]),
        ]
    )
    updates_list = [updates_1, updates_2, updates_3]
    return {
        f"client_{idx + 1}": TrainReply(
            n_epoch=1,
            n_steps=10,
            t_spent=0,
            updates=ModelUpdates(updates_list[idx], weights=1),
            aux_var={},
        )
        for idx in range(n_clients)
    }


@pytest.fixture(name="server_model")
def server_model_fixture(
    framework: FrameworkType,
) -> Model:
    """
    Build a mock Model specifically for client sampler unit tests.

    Only the get_weights method is implemented, as it is the sole requirement
    for these tests. When called, it returns a framework-specific weight vector
    containing constant values.
    """
    model = MagicMock(spec=Model)

    def get_weights_mock(trainable=False):
        return GradientsTestCase(framework).mock_gradient_custom(
            [
                np.array([2, 2, 2, 2]),
                np.array([2, 2]),
            ]
        )

    model.get_weights.side_effect = get_weights_mock
    return model
