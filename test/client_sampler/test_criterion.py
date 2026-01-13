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

import math
from typing import Dict

import pytest

from declearn.client_sampler.modules import (
    GradientNormCriterion,
    NormalizedDivCriterion,
)
from declearn.messaging import TrainReply
from declearn.model.api import Model
from declearn.test_utils import list_available_frameworks

VECTOR_FRAMEWORKS = list_available_frameworks()


class TestCriterion:
    """Shared unit tests suite for 'Criterion' subclasses."""

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_gradient_norm_criterion(
        self,
        client_to_reply: Dict[str, TrainReply],
    ) -> None:
        criterion = GradientNormCriterion()

        expected_scores = {
            "client_1": 0,
            "client_2": math.sqrt(6),
            "client_3": math.sqrt(10.25),
        }

        scores = criterion.compute(client_to_reply, None)
        for client in scores:
            assert math.isclose(
                expected_scores[client], scores[client], rel_tol=1e-6
            )

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_composition_criterion(
        self,
        client_to_reply: Dict[str, TrainReply],
    ) -> None:
        criterion = GradientNormCriterion() ** 2 / 2

        expected_scores = {
            "client_1": 0,
            "client_2": math.sqrt(6) ** 2 / 2,
            "client_3": math.sqrt(10.25) ** 2 / 2,
        }

        scores = criterion.compute(client_to_reply, None)
        for client in scores:
            assert math.isclose(
                expected_scores[client], scores[client], rel_tol=1e-6
            )

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_normalized_div_criterion(
        self,
        client_to_reply: Dict[str, TrainReply],
        server_model: Model,
    ) -> None:
        criterion = NormalizedDivCriterion()

        expected_scores = {
            "client_1": 0,
            "client_2": 1 / 2,
            "client_3": 13 / 24,
        }

        scores = criterion.compute(client_to_reply, server_model)
        for client in scores:
            assert math.isclose(
                expected_scores[client], scores[client], rel_tol=1e-6
            )
