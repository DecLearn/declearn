"""Shared pytest fixtures for client sampler testing."""

from typing import Dict

import pytest

from declearn.aggregator import ModelUpdates
from declearn.messaging import TrainReply
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
)
from declearn.utils import set_device_policy


@pytest.fixture(name="train_replies")
def train_replies_fixture(
    framework: FrameworkType,
    n_clients: int = 3,
) -> Dict[str, TrainReply]:
    """Build the dictionary of client replies (messages) from the updates."""
    set_device_policy(gpu=False)
    return {
        str(idx): TrainReply(
            n_epoch=1,
            n_steps=10,
            t_spent=0,
            updates=ModelUpdates(
                GradientsTestCase(framework, seed=idx).mock_gradient, weights=1
            ),
            aux_var={},
        )
        for idx in range(n_clients)
    }
