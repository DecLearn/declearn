# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
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

"""Unit tests for `declearn.main.utils.TrainingManager`."""

from typing import Any, Iterator, Optional
from unittest import mock

import numpy

from declearn import messaging
from declearn.aggregator import Aggregator
from declearn.dataset import Dataset
from declearn.metrics import Metric, MetricSet
from declearn.model.api import Model, Vector
from declearn.optimizer import Optimizer
from declearn.training import TrainingManager

MockArray = mock.create_autospec(numpy.ndarray)
MockVector = mock.create_autospec(Vector)
BATCHES = {"batch_size": 42}  # default batch-generation kwargs


def build_manager(n_batch: int) -> Any:  # TrainingManager with Mock attributes
    """Return a TrainingManager instance with Mock attributes."""
    model = mock.create_autospec(Model, instance=True)
    model.compute_batch_predictions.return_value = (MockArray, MockArray, None)
    optim = mock.create_autospec(Optimizer, instance=True)
    aggrg = mock.create_autospec(Aggregator, instance=True)
    train_data = build_mock_dataset(n_batch)
    valid_data = build_mock_dataset(n_batch)
    metrics = mock.create_autospec(MetricSet, instance=True)
    metrics.metrics = []
    return TrainingManager(
        model, optim, aggrg, train_data, valid_data, metrics
    )


def build_mock_dataset(n_batch: int) -> Dataset:
    """Build a Mock Dataset that yields a fixed number of batches."""

    def get_mock_batches(**_) -> Iterator[Any]:
        """Yield mock data batches."""
        nonlocal n_batch
        return ((mock.MagicMock(), None, None) for _ in range(n_batch))

    dataset = mock.create_autospec(Dataset, instance=True)
    dataset.generate_batches.side_effect = get_mock_batches
    return dataset


def build_train_request(
    n_epoch: Optional[int] = None,
    n_steps: Optional[int] = None,
    timeout: Optional[float] = None,
) -> messaging.TrainRequest:
    """Return a TrainRequest with specified constraint parameters."""
    return messaging.TrainRequest(
        round_i=0,
        weights=MockVector({}),
        aux_var={},
        batches=BATCHES,
        n_epoch=n_epoch,
        n_steps=n_steps,
        timeout=timeout,  # type: ignore
    )


class TestTrainingRound:
    """Unit tests for `declearn.main.utils.TrainingManager.training_round`.

    These tests verify that the `training_round` routine
    * can be called with a variety of constraint specifications
    * results in the specified number of epochs and/or steps to be run
    * reports accurate values for the number of epochs and steps taken
    """

    def test_training_round_with_epoch_constraint(self) -> None:
        """Test running a 1-epoch training round."""
        manager = build_manager(n_batch=100)
        reply = manager.training_round(build_train_request(n_epoch=1))
        assert isinstance(reply, messaging.TrainReply)
        assert reply.n_epoch == 1
        assert reply.n_steps == 100
        assert manager.optim.run_train_step.call_count == 100
        manager.train_data.generate_batches.assert_called_once_with(**BATCHES)
        manager.aggrg.prepare_for_sharing.assert_called_once_with(
            updates=mock.ANY, n_steps=100
        )

    def test_training_round_with_steps_constraint_1(self) -> None:
        """Test running a 20-steps (< 1 epoch) training round."""
        manager = build_manager(n_batch=100)
        reply = manager.training_round(build_train_request(n_steps=20))
        assert isinstance(reply, messaging.TrainReply)
        assert reply.n_epoch == 1
        assert reply.n_steps == 20
        assert manager.optim.run_train_step.call_count == 20
        manager.train_data.generate_batches.assert_called_once_with(**BATCHES)
        manager.aggrg.prepare_for_sharing.assert_called_once_with(
            updates=mock.ANY, n_steps=20
        )

    def test_training_round_with_steps_constraint_2(self) -> None:
        """Test running a 150-steps (1.5 epochs) training round."""
        manager = build_manager(n_batch=100)
        reply = manager.training_round(build_train_request(n_steps=150))
        assert isinstance(reply, messaging.TrainReply)
        assert reply.n_epoch == 2
        assert reply.n_steps == 150
        assert manager.optim.run_train_step.call_count == 150
        assert manager.train_data.generate_batches.call_count == 2
        manager.aggrg.prepare_for_sharing.assert_called_once_with(
            updates=mock.ANY, n_steps=150
        )

    def test_training_round_with_timeout_constraint(self) -> None:
        """Test running a time-constrained training round."""
        manager = build_manager(n_batch=100)
        reply = manager.training_round(build_train_request(timeout=0.1))
        assert isinstance(reply, messaging.TrainReply)
        assert 0.1 <= reply.t_spent
        assert manager.optim.run_train_step.call_count == reply.n_steps
        assert manager.train_data.generate_batches.call_count == reply.n_epoch
        manager.aggrg.prepare_for_sharing.assert_called_once_with(
            updates=mock.ANY, n_steps=reply.n_steps
        )

    def test_training_round_with_multiple_constraints_1(self) -> None:
        """Test running a min(3 epoch, 150 steps) training round."""
        manager = build_manager(n_batch=100)
        request = build_train_request(n_epoch=3, n_steps=150)
        reply = manager.training_round(request)
        assert isinstance(reply, messaging.TrainReply)
        assert reply.n_epoch == 2
        assert reply.n_steps == 150
        assert manager.optim.run_train_step.call_count == 150
        assert manager.train_data.generate_batches.call_count == 2
        manager.aggrg.prepare_for_sharing.assert_called_once_with(
            updates=mock.ANY, n_steps=150
        )

    def test_training_round_with_multiple_constraints_2(self) -> None:
        """Test running a min(3 epoch, 500 steps) training round."""
        manager = build_manager(n_batch=100)
        request = build_train_request(n_epoch=3, n_steps=500)
        reply = manager.training_round(request)
        assert isinstance(reply, messaging.TrainReply)
        assert reply.n_epoch == 3
        assert reply.n_steps == 300
        assert manager.optim.run_train_step.call_count == 300
        assert manager.train_data.generate_batches.call_count == 3
        manager.aggrg.prepare_for_sharing.assert_called_once_with(
            updates=mock.ANY, n_steps=300
        )


def build_evaluation_request(
    n_steps: Optional[int] = None,
    timeout: Optional[float] = None,
) -> messaging.EvaluationRequest:
    """Return an EvaluationRequest with specified constraint parameters."""
    return messaging.EvaluationRequest(
        round_i=0,
        weights=MockVector({}),
        batches=BATCHES,
        n_steps=n_steps,
        timeout=timeout,  # type: ignore
    )


class TestEvaluationRound:
    """Unit tests for `declearn.main.utils.TrainingManager.evaluation_round`.

    These tests verify that the `evaluation_round` routine
    * can be called with a variety of constraint specifications
    * results in the specified number of steps to be run
    * reports accurate values for the number of steps taken
    """

    def test_metrics_instantiation(self) -> None:
        """Test that the model's loss is added to the wrapped metrics."""
        manager = build_manager(n_batch=1)
        assert len(manager.metrics.metrics) == 1
        assert isinstance(manager.metrics.metrics[0], Metric)
        assert manager.metrics.metrics[0].name == "loss"

    def test_evaluation_round_without_constraints(self) -> None:
        """Test running a 1-epoch evaluation round."""
        manager = build_manager(n_batch=100)
        reply = manager.evaluation_round(build_evaluation_request())
        assert isinstance(reply, messaging.EvaluationReply)
        assert reply.n_steps == 100
        assert manager.metrics.update.call_count == 100
        manager.metrics.get_result.assert_called_once()
        assert reply.metrics == manager.metrics.get_states.return_value
        manager.valid_data.generate_batches.assert_called_once_with(**BATCHES)

    def test_evaluation_round_with_steps_constraint(self) -> None:
        """Test running a 50-steps (half-epoch) evaluation round."""
        manager = build_manager(n_batch=100)
        reply = manager.evaluation_round(build_evaluation_request(n_steps=50))
        assert isinstance(reply, messaging.EvaluationReply)
        assert reply.n_steps == 50
        assert manager.metrics.update.call_count == 50
        manager.metrics.get_result.assert_called_once()
        assert reply.metrics == manager.metrics.get_states.return_value
        manager.valid_data.generate_batches.assert_called_once_with(**BATCHES)

    def test_evaluation_round_with_loose_steps_constraint(self) -> None:
        """Test running an evaluation round with a loose steps constraint."""
        manager = build_manager(n_batch=100)
        reply = manager.evaluation_round(build_evaluation_request(n_steps=150))
        assert isinstance(reply, messaging.EvaluationReply)
        assert reply.n_steps == 100
        assert manager.metrics.update.call_count == 100
        manager.metrics.get_result.assert_called_once()
        assert reply.metrics == manager.metrics.get_states.return_value
        manager.valid_data.generate_batches.assert_called_once_with(**BATCHES)

    def test_evaluation_round_with_timeout_constraint(self) -> None:
        """Test running an evaluation round with a 0.1 second constraint."""
        manager = build_manager(n_batch=10000)
        reply = manager.evaluation_round(build_evaluation_request(timeout=0.1))
        assert isinstance(reply, messaging.EvaluationReply)
        assert reply.n_steps < 10000
        assert 0.1 <= reply.t_spent
        assert manager.metrics.update.call_count == reply.n_steps
        manager.metrics.get_result.assert_called_once()
        assert reply.metrics == manager.metrics.get_states.return_value
        manager.valid_data.generate_batches.assert_called_once_with(**BATCHES)

    def test_evaluation_round_with_multiple_constraints_1(self) -> None:
        """Test running an evaluation round for min(50 steps, 20 seconds)."""
        manager = build_manager(n_batch=100)
        request = build_evaluation_request(n_steps=50, timeout=20)
        reply = manager.evaluation_round(request)
        assert isinstance(reply, messaging.EvaluationReply)
        assert reply.n_steps == 50
        assert reply.t_spent < 20
        assert manager.metrics.update.call_count == 50
        manager.metrics.get_result.assert_called_once()
        assert reply.metrics == manager.metrics.get_states.return_value
        manager.valid_data.generate_batches.assert_called_once_with(**BATCHES)

    def test_evaluation_round_with_multiple_constraints_2(self) -> None:
        """Test running an evaluation round for min(10k steps, 0.1 second)."""
        manager = build_manager(n_batch=12000)
        request = build_evaluation_request(n_steps=10000, timeout=0.1)
        reply = manager.evaluation_round(request)
        assert isinstance(reply, messaging.EvaluationReply)
        assert reply.n_steps < 10000
        assert 0.1 <= reply.t_spent
        assert manager.metrics.update.call_count == reply.n_steps
        manager.metrics.get_result.assert_called_once()
        assert reply.metrics == manager.metrics.get_states.return_value
        manager.valid_data.generate_batches.assert_called_once_with(**BATCHES)
