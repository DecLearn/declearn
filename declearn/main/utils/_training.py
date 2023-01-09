# coding: utf-8

"""Wrapper to run local training and evaluation rounds in a FL process."""

import logging
from typing import Any, Dict, Optional, Union

from declearn.communication import messaging
from declearn.dataset import Dataset
from declearn.main.utils._constraints import (
    Constraint,
    ConstraintSet,
    TimeoutConstraint,
)
from declearn.model.api import Model
from declearn.optimizer import Optimizer
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
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        self.model = model
        self.optim = optim
        self.train_data = train_data
        self.valid_data = valid_data
        if not isinstance(logger, logging.Logger):
            logger = get_logger(logger or f"{type(self).__name__}")
        self.logger = logger

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
        self.model.set_weights(message.weights)
        self.optim.process_aux_var(message.aux_var)
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
            updates=message.weights - self.model.get_weights(),
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
        while not (constraints.saturated or epochs.saturated):
            for batch in self.train_data.generate_batches(**batch_cfg):
                self.optim.run_train_step(self.model, batch)
                constraints.increment()
                if constraints.saturated:
                    break
            epochs.increment()
        # Return a dict storing information on the training effort.
        effort = {"n_epoch": epochs.value}
        effort.update(constraints.get_values())
        return effort

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
            self.model.set_weights(message.weights)  # revise: optional
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
        # Run batch evaluation steps for as long as constraints allow it.
        loss = 0.0
        dataset = self.valid_data or self.train_data
        for batch in dataset.generate_batches(**batch_cfg):
            loss += self.model.compute_loss([batch])
            constraints.increment()
            if constraints.saturated:
                break
        # Pack the result and computational effort information into a message.
        effort = constraints.get_values()
        return messaging.EvaluationReply(
            loss=loss / effort["n_steps"],
            n_steps=int(effort["n_steps"]),
            t_spent=round(effort["t_spent"], 3),
        )