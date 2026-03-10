# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Shared pytest fixtures for client sampler testing."""

from typing import Dict, Set
from unittest.mock import MagicMock

import numpy as np
import pytest

from declearn.aggregator import ModelUpdates
from declearn.messaging import TrainReply
from declearn.model.api import Model
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
)


@pytest.fixture
def clients() -> Set[str]:
    return set(["client1", "client2", "client3"])


@pytest.fixture(name="client_to_reply")
def client_to_reply_fixture(
    framework: FrameworkType,
    n_clients: int = 3,
) -> Dict[str, TrainReply]:
    """
    Build a dictionary of client replies (messages) from updates that we define
    as vectors with custom value.
    """
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
    t_spent_list = [10.0, 20.0, 30.0]
    return {
        f"client_{idx + 1}": TrainReply(
            n_epoch=1,
            n_steps=10,
            t_spent=t_spent_list[idx],
            updates=ModelUpdates(updates_list[idx], weights=1),
            aux_var={},
        )
        for idx in range(n_clients)
    }


@pytest.fixture(name="global_model")
def global_model_fixture(
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
