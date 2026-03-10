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

"""Unit tests for the 'Criterion' subclasses."""

import math
from typing import Dict

import pytest

from declearn.aggregator import ModelUpdates
from declearn.client_sampler.criterion import (
    CompositionCriterion,
    ConstantCriterion,
    GradientNormCriterion,
    NormalizedDivCriterion,
    TrainTimeCriterion,
    TrainTimeHistoryCriterion,
)
from declearn.messaging import TrainReply
from declearn.model.api import Model
from declearn.test_utils import GradientsTestCase, list_available_frameworks

VECTOR_FRAMEWORKS = list_available_frameworks()


class TestCriterion:
    """Shared unit tests suite for 'Criterion' subclasses."""

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_gradient_norm_criterion(
        self,
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
    ) -> None:
        criterion = GradientNormCriterion()

        expected_scores = {
            "client_1": 0,
            "client_2": math.sqrt(6),
            "client_3": math.sqrt(10.25),
        }

        scores = criterion.compute(client_to_reply, global_model)
        for client in scores:
            assert math.isclose(
                expected_scores[client], scores[client], rel_tol=1e-6
            )

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_composition_criterion(
        self,
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
    ) -> None:
        criterion = GradientNormCriterion() ** 2 / 2

        expected_scores = {
            "client_1": 0,
            "client_2": math.sqrt(6) ** 2 / 2,
            "client_3": math.sqrt(10.25) ** 2 / 2,
        }

        scores = criterion.compute(client_to_reply, global_model)
        for client in scores:
            assert math.isclose(
                expected_scores[client], scores[client], rel_tol=1e-6
            )

    def test_composition_criterion_invalid(
        self,
    ) -> None:
        criterion1 = GradientNormCriterion()
        criterion2 = ConstantCriterion(1)
        with pytest.raises(ValueError):
            _ = CompositionCriterion("my_operator", criterion1, criterion2)

    @pytest.mark.parametrize("framework", VECTOR_FRAMEWORKS)
    def test_normalized_div_criterion(
        self,
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
    ) -> None:
        criterion = NormalizedDivCriterion()

        expected_scores = {
            "client_1": 0,
            "client_2": 1 / 2,
            "client_3": 13 / 24,
        }

        scores = criterion.compute(client_to_reply, global_model)
        for client in scores:
            assert math.isclose(
                expected_scores[client], scores[client], rel_tol=1e-6
            )

    @pytest.mark.parametrize("framework", ["torch"])
    def test_train_time_criterion_lowest(
        self,
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
    ) -> None:
        criterion = TrainTimeCriterion()

        expected_scores = {
            "client_1": -10.0,
            "client_2": -20.0,
            "client_3": -30.0,
        }

        scores = criterion.compute(client_to_reply, global_model)
        for client in scores:
            assert math.isclose(
                expected_scores[client], scores[client], rel_tol=1e-6
            )

    @pytest.mark.parametrize("framework", ["torch"])
    def test_train_time_criterion_highest(
        self,
        client_to_reply: Dict[str, TrainReply],
        global_model: Model,
    ) -> None:
        criterion = TrainTimeCriterion(lower_is_better=False)

        expected_scores = {
            "client_1": 10.0,
            "client_2": 20.0,
            "client_3": 30.0,
        }

        scores = criterion.compute(client_to_reply, global_model)
        for client in scores:
            assert math.isclose(
                expected_scores[client], scores[client], rel_tol=1e-6
            )

    @pytest.mark.parametrize("agg", ["average", "sum"])
    @pytest.mark.parametrize("framework", ["torch"])
    def test_train_time_hist_criterion_lowest(
        self,
        agg: TrainTimeHistoryCriterion.AggregateFunc,
        global_model: Model,
    ) -> None:
        DEFAULT_EPOCHS = 1
        DEFAULT_STEPS = 10
        DEFAULT_GRAD = ModelUpdates(
            GradientsTestCase("torch").mock_ones,
            weights=1,
        )

        criterion = TrainTimeHistoryCriterion(
            lower_is_better=True,
            agg=agg,
        )

        # 1st fake round results
        client_to_reply = {
            "client_1": TrainReply(
                n_epoch=DEFAULT_EPOCHS,
                n_steps=DEFAULT_STEPS,
                t_spent=20.0,
                updates=DEFAULT_GRAD,
                aux_var={},
            ),
            "client_2": TrainReply(
                n_epoch=DEFAULT_EPOCHS,
                n_steps=DEFAULT_STEPS,
                t_spent=10.0,
                updates=DEFAULT_GRAD,
                aux_var={},
            ),
        }
        # compute score after fake round 1
        scores = criterion.compute(client_to_reply, global_model)

        # fake round 2, update only the training time
        client_to_reply = {
            "client_2": TrainReply(
                n_epoch=DEFAULT_EPOCHS,
                n_steps=DEFAULT_STEPS,
                t_spent=40.0,
                updates=DEFAULT_GRAD,
                aux_var={},
            ),
            "client_3": TrainReply(
                n_epoch=DEFAULT_EPOCHS,
                n_steps=DEFAULT_STEPS,
                t_spent=30.0,
                updates=DEFAULT_GRAD,
                aux_var={},
            ),
        }
        # compute score after fake round 2
        scores = criterion.compute(client_to_reply, global_model)

        # fake round 3, update sampled clients and training time
        client_to_reply = {
            "client_1": TrainReply(
                n_epoch=DEFAULT_EPOCHS,
                n_steps=DEFAULT_STEPS,
                t_spent=50.0,
                updates=DEFAULT_GRAD,
                aux_var={},
            ),
            "client_2": TrainReply(
                n_epoch=DEFAULT_EPOCHS,
                n_steps=DEFAULT_STEPS,
                t_spent=70.0,
                updates=DEFAULT_GRAD,
                aux_var={},
            ),
            "client_3": TrainReply(
                n_epoch=DEFAULT_EPOCHS,
                n_steps=DEFAULT_STEPS,
                t_spent=60.0,
                updates=DEFAULT_GRAD,
                aux_var={},
            ),
        }
        # compute score after fake round 3
        scores = criterion.compute(client_to_reply, global_model)

        expected_histories = {
            "client_1": [20.0, 50.0],
            "client_2": [10.0, 40.0, 70.0],
            "client_3": [30.0, 60.0],
        }

        agg_to_expected_scores = {
            "average": {
                "client_1": -35.0,
                "client_2": -40.0,
                "client_3": -45.0,
            },
            "sum": {
                "client_1": -70.0,
                "client_2": -120.0,
                "client_3": -90.0,
            },
        }

        # check history is correct
        for client, expected_history in expected_histories.items():
            history = criterion.history[client]
            assert len(expected_history) == len(history)
            for expected, value in zip(expected_history, history, strict=True):
                assert math.isclose(expected, value, rel_tol=1e-6)

        # check score is correctly computed from history
        for client, expected_score in agg_to_expected_scores[agg].items():
            assert math.isclose(expected_score, scores[client], rel_tol=1e-6)
