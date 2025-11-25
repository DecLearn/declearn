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

"""Unit tests for the 'Criterion' subclasses."""

from typing import Dict

import numpy as np
import pytest

from declearn.aggregator import ModelUpdates
from declearn.client_sampler.modules import GradientNormCriterion
from declearn.messaging import TrainReply
from declearn.test_utils import (
    FrameworkType,
    GradientsTestCase,
    list_available_frameworks,
)
from declearn.utils import set_device_policy

VECTOR_FRAMEWORKS = list_available_frameworks()


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


class TestCriterion:
    """Shared unit tests suite for 'Criterion' subclasses."""

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_gradient_criterion(
        self,
        train_replies: Dict[str, TrainReply],
    ) -> None:
        """"""
        criterion = GradientNormCriterion()

        expected_values = {}
        for client_name, client_reply in train_replies.items():
            # Compute expected values
            client_gradients, _ = client_reply.updates.updates.flatten()
            expected_values[client_name] = np.linalg.norm(client_gradients)

        criterion_values = criterion.compute(train_replies)
        assert criterion_values == expected_values

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_composition_criterion(
        self,
        train_replies: Dict[str, TrainReply],
    ) -> None:
        criterion = GradientNormCriterion() ** 2 / 2

        expected_values = {}
        for client_name, client_reply in train_replies.items():
            # Compute expected values
            client_gradients, _ = client_reply.updates.updates.flatten()
            expected_values[client_name] = (
                np.linalg.norm(client_gradients) ** 2 / 2
            )

        criterion_values = criterion.compute(train_replies)
        assert criterion_values == expected_values
