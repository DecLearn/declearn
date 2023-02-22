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

"""Wrapper to run local training and evaluation rounds in a FL process."""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Union

import numpy as np

from declearn.communication import messaging
from declearn.dataset import Dataset
from declearn.main.utils._constraints import (
    Constraint,
    ConstraintSet,
    TimeoutConstraint,
)
from declearn.metrics import MeanMetric, Metric, MetricInputType, MetricSet
from declearn.model.api import Model
from declearn.optimizer import Optimizer
from declearn.typing import Batch
from declearn.utils import get_logger

__all__ = [
    "TrainingManager",
]


class TrainingManager:
    """Class wrapping the logic for local training and evaluation rounds."""

    def __init__(
        self,
        model: Model,
        optim: Optimizer,
        train_data: Dataset,
        valid_data: Optional[Dataset] = None,
        metrics: Union[MetricSet, List[MetricInputType], None] = None,
        logger: Union[logging.Logger, str, None] = None,
    ) -> None:
        """Instantiate the client-side training and evaluation process.

        Arguments
        ---------
        model: Model
            Model instance that needs training and/or evaluating.
        optim: Optimizer
            Optimizer instance that orchestrates training steps.
        train_data: Dataset
            Dataset instance wrapping the local training dataset.
        valid_data: Dataset or None, default=None
            Dataset instance wrapping the local validation dataset.
            If None, use `train_data` in the evaluation rounds.
        metrics: MetricSet or list[MetricInputType] or None, default=None
            MetricSet instance or list of Metric instances and/or specs
            to wrap into one, defining evaluation metrics to compute in
            addition to the model's loss.
            If None, only compute and report the model's loss.
        logger: logging.Logger or str or None, default=None,
            Logger to use, or name of a logger to set up with
            `declearn.utils.get_logger`.
            If None, use `type(self).__name__`.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        self.model = model
        self.optim = optim
        self.train_data = train_data
        self.valid_data = valid_data
        self.metrics = self._prepare_metrics(metrics)
        if not isinstance(logger, logging.Logger):
            logger = get_logger(logger or f"{type(self).__name__}")
        self.logger = logger

    def _prepare_metrics(
        self,
        metrics: Union[MetricSet, List[MetricInputType], None],
    ) -> MetricSet:
        """Parse the `metrics` instantiation inputs into a MetricSet."""
        # Type-check and/or transform the inputs into a MetricSet instance.
        metrics = MetricSet.from_specs(metrics)
        # If a model loss metric is part of the set, remove it.
        for i, metric in enumerate(metrics.metrics):
            if metric.name == "loss":
                metrics.metrics.pop(i)
        # Add the wrapped model's loss to the metrics.
        loss = self._setup_loss_metric()
        metrics.metrics.append(loss)
        # Return the prepared object for assignment as `metrics` attribute.
        return metrics

    def _setup_loss_metric(
        self,
    ) -> Metric:
        """Return an ad-hoc Metric object to compute the model's loss."""
        loss_fn = self.model.loss_function

        # Write a custom, unregistered Metric subclass.
        class LossMetric(MeanMetric, register=False):
            """Ad hoc Metric wrapping a model's loss function."""

            name: ClassVar[str] = "loss"

            def metric_func(
                self, y_true: np.ndarray, y_pred: np.ndarray
            ) -> np.ndarray:
                return loss_fn(y_true, y_pred)

        # Instantiate and return the ad-hoc loss metric.
        return LossMetric()

    def training_round(
        self,
        message: messaging.TrainRequest,
    ) -> Union[messaging.TrainReply, messaging.Error]:
        """Run a local training round.

        If an exception is raised during the local process, wrap it as
        an Error message instead of raising it.

        Parameters
        ----------
        message: TrainRequest
            Instructions from the server regarding the training round.

        Returns
        -------
        reply: TrainReply or Error
            Message wrapping results from the training round, or any
            error raised during it.
        """
        self.logger.info("Participating in training round %s", message.round_i)
        # Try running the training round; return the reply is successful.
        try:
            return self._training_round(message)
        # In case of failure, wrap the exception as an Error message.
        except Exception as exception:  # pylint: disable=broad-except
            self.logger.error(
                "Error encountered during training: %s.", exception
            )
            return messaging.Error(repr(exception))

    def _training_round(
        self,
        message: messaging.TrainRequest,
    ) -> messaging.TrainReply:
        """Backend to `training_round`, without exception capture hooks."""
        # Unpack and apply model weights and optimizer auxiliary variables.
        self.logger.info("Applying server updates to local objects.")
        self.model.set_weights(message.weights, trainable=True)
        self.optim.process_aux_var(message.aux_var)
        self.optim.start_round()  # trigger loss regularizer's `on_round_start`
        # Train under instructed effort constraints.
        params = message.n_epoch, message.n_steps, message.timeout
        self.logger.info(
            "Training local model for %s epochs | %s steps | %s seconds.",
            *params,
        )
        effort = self._train_under_constraints(message.batches, *params)
        # Compute model updates and collect auxiliary variables.
        self.logger.info("Packing local updates to be sent to the server.")
        return messaging.TrainReply(
            updates=message.weights - self.model.get_weights(trainable=True),
            aux_var=self.optim.collect_aux_var(),
            n_epoch=int(effort["n_epoch"]),
            n_steps=int(effort["n_steps"]),
            t_spent=round(effort["t_spent"], 3),
        )

    def _train_under_constraints(
        self,
        batch_cfg: Dict[str, Any],
        n_epoch: Optional[int],
        n_steps: Optional[int],
        timeout: Optional[int],
    ) -> Dict[str, float]:
        """Backend code to run local SGD steps under effort constraints.

        Parameters
        ----------
        batch_cfg: Dict[str, Any]
            Keyword arguments for `self.train_data.generate_batches`
            i.e. specifications of batches used in local SGD steps.
        n_epoch: int or None, default=None
            Maximum number of local training epochs to perform.
            May be overridden by `n_steps` or `timeout`.
        n_steps: int or None, default=None
            Maximum number of local training steps to perform.
            May be overridden by `n_epoch` or `timeout`.
        timeout: int or None, default=None
            Time (in seconds) beyond which to interrupt training,
            regardless of the actual number of steps taken (> 0).

        Returns
        -------
        effort: dict[str, float]
            Dictionary storing information on the computational
            effort effectively performed:
            * n_epoch: int
                Number of training epochs completed.
            * n_steps: int
                Number of training steps completed.
            * t_spent: float
                Time spent running training steps (in seconds).
        """
        # Set up effort constraints under which to operate.
        epochs = Constraint(limit=n_epoch, name="n_epoch")
        constraints = ConstraintSet(
            Constraint(limit=n_steps, name="n_steps"),
            TimeoutConstraint(limit=timeout, name="t_spent"),
        )
        # Run batch train steps for as long as constraints allow it.
        stop_training = False
        while not (stop_training or epochs.saturated):
            for batch in self.train_data.generate_batches(**batch_cfg):
                try:
                    self._run_train_step(batch)
                except StopIteration as exc:
                    self.logger.warning("Interrupting training round: %s", exc)
                    stop_training = True
                    break
                constraints.increment()
                if constraints.saturated:
                    stop_training = True
                    break
            epochs.increment()
        # Return a dict storing information on the training effort.
        effort = {"n_epoch": epochs.value}
        effort.update(constraints.get_values())
        return effort

    def _run_train_step(
        self,
        batch: Batch,
    ) -> None:
        """Run a single training step based on an input batch.

        Parameters
        ----------
        batch: Batch
            Batched data based on which to compute and apply model updates.

        Raises
        ------
        StopIteration:
            If this step is being cancelled and the training routine
            in the context of which it is being called should stop.
        """
        self.optim.run_train_step(self.model, batch)

    def evaluation_round(
        self,
        message: messaging.EvaluationRequest,
    ) -> Union[messaging.EvaluationReply, messaging.Error]:
        """Run a local evaluation round.

        If an exception is raised during the local process, wrap it as
        an Error message instead of raising it.

        Parameters
        ----------
        message: EvaluationRequest
            Instructions from the server regarding the evaluation round.

        Returns
        -------
        reply: EvaluationReply or Error
            Message wrapping results from the evaluation round, or any
            error raised during it.
        """
        self.logger.info(
            "Participating in evaluation round %s", message.round_i
        )
        # Try running the evaluation round.
        try:
            # Update the model's weights and evaluate on the local dataset.
            # Revise: make the weights' update optional.
            self.model.set_weights(message.weights, trainable=True)
            return self._evaluate_under_constraints(
                message.batches, message.n_steps, message.timeout
            )
        # In case of failure, wrap the exception as an Error message.
        except Exception as exception:  # pylint: disable=broad-except
            self.logger.error(
                "Error encountered during evaluation: %s.", exception
            )
            return messaging.Error(repr(exception))

    def _evaluate_under_constraints(
        self,
        batch_cfg: Dict[str, Any],
        n_steps: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> messaging.EvaluationReply:
        """Backend code to run local loss computation under effort constraints.

        Parameters
        ----------
        batch_cfg: Dict[str, Any]
            Keyword arguments to `self.valid_data.generate_batches`.
        n_steps: int or None, default=None
            Maximum number of local evaluation steps to perform.
            May be overridden by `timeout` or dataset size.
        timeout: int or None, default=None
            Time (in seconds) beyond which to interrupt evaluation,
            regardless of the actual number of steps taken (> 0).

        Returns
        -------
        reply: messaging.EvaluationReply
            EvaluationReply message wrapping the computed loss on the
            local validation (or, if absent, training) dataset as well
            as the number of steps and the time taken to obtain it.
        """
        # Set up effort constraints under which to operate.
        constraints = ConstraintSet(
            Constraint(limit=n_steps, name="n_steps"),
            TimeoutConstraint(limit=timeout, name="t_spent"),
        )
        # Ensure evaluation metrics start from their base state.
        self.metrics.reset()
        # Run batch evaluation steps for as long as constraints allow it.
        dataset = self.valid_data or self.train_data
        for batch in dataset.generate_batches(**batch_cfg):
            inputs = self.model.compute_batch_predictions(batch)
            self.metrics.update(*inputs)
            constraints.increment()
            if constraints.saturated:
                break
        # Gather the computed metrics and computational effort information.
        effort = constraints.get_values()
        result = self.metrics.get_result()
        states = self.metrics.get_states()
        self.logger.info(
            "Local scalar evaluation metrics: %s",
            {k: v for k, v in result.items() if isinstance(v, float)},
        )
        # Pack the result and computational effort information into a message.
        self.logger.info("Packing local results to be sent to the server.")
        return messaging.EvaluationReply(
            loss=float(result["loss"]),
            metrics=states,
            n_steps=int(effort["n_steps"]),
            t_spent=round(effort["t_spent"], 3),
        )
