# coding: utf-8

"""Dataclasses to wrap and parse some training-related hyperparameters."""

import dataclasses

from typing import Any, Dict, Optional


__all__ = [
    "EvaluateConfig",
    "RegisterConfig",
    "TrainingConfig",
]


@dataclasses.dataclass
class RegisterConfig:
    """Dataclass wrapping parameters for clients' registration.

    The parameters wrapped by this class are those of
    `declearn.communication.Server.wait_for_clients`.

    Attributes
    ----------
    min_clients: int
        Minimum number of clients required. Corrected to be >= 1.
        If `timeout` is None, used as the exact number of clients
        required - once reached, registration will be closed.
    max_clients: int or None
        Maximum number of clients authorized to register.
    timeout: int or None
        Optional maximum waiting time (in seconds) beyond which
        to close registration and either return or raise.
    """

    min_clients: int = 1
    max_clients: Optional[int] = None
    timeout: Optional[int] = None


@dataclasses.dataclass
class TrainingConfig:
    """Dataclass wrapping parameters for a training round.

    The parameters wrapped by this class are those
    of `declearn.dataset.Dataset.generate_batches`
    and `declearn.main.FederatedClient._train_for`.

    Attributes
    ----------
    batch_size: int
        Number of samples per processed data batch.
    shuffle: bool
        Whether to shuffle data samples prior to batching.
    drop_remainder: bool
        Whether to drop the last batch if it contains less
        samples than `batch_size`, or yield it anyway.
    n_epoch: int or None
        Maximum number of local data-processing epochs to
        perform. May be overridden by `n_steps` or `timeout`.
    n_steps: int or None
        Maximum number of local data-processing steps to
        perform. May be overridden by `n_epoch` or `timeout`.
    timeout: int or None
        Time (in seconds) beyond which to interrupt processing,
        regardless of the actual number of steps taken (> 0).
    """

    # Dataset.generate_batches() parameters
    batch_size: int
    shuffle: bool = False
    drop_remainder: bool = True
    # training effort constraints
    n_epoch: Optional[int] = 1
    n_steps: Optional[int] = None
    timeout: Optional[int] = None

    def __post_init__(self) -> None:
        if all(v is None for v in (self.n_epoch, self.n_steps, self.timeout)):
            raise ValueError(
                "At least one effort constraint must be set: "
                "n_epoch, n_steps and timeout cannot all be None."
            )

    @property
    def batch_cfg(self) -> Dict[str, Any]:
        """Batches-generation parameters from this config."""
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_remainder": self.drop_remainder,
        }

    @property
    def message_params(self) -> Dict[str, Any]:
        """TrainRequest message parameters from this config."""
        return {
            "batches": self.batch_cfg,
            "n_epoch": self.n_epoch,
            "n_steps": self.n_steps,
            "timeout": self.timeout,
        }


@dataclasses.dataclass
class EvaluateConfig(TrainingConfig):
    """Dataclass wrapping parameters for an evaluation round.

    Please refer to the parent class `TrainingConfig` for details
    on the wrapped parameters / attribute. Note that `n_epoch` is
    most-likely ignored by handling classes.
    """

    drop_remainder: bool = False

    @property
    def message_params(self) -> Dict[str, Any]:
        """ValidRequest message parameters from this config."""
        params = super().message_params
        params.pop("n_epoch")
        return params
