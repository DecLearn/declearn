# coding: utf-8

"""Dataclasses to wrap and parse some training-related hyperparameters."""

import dataclasses

from typing import Any, Dict, Optional


@dataclasses.dataclass
class TrainingConfig:
    """Dataclass wrapping parameters for a training round."""

    # Dataset.generate_batches() parameters
    batch_size: int
    shuffle: bool = False
    seed: Optional[int] = None
    drop_remainder: bool = True
    # training effort constraints
    n_epoch: Optional[int] = 1
    n_steps: Optional[int] = None
    timeout: Optional[int] = None

    @property
    def batch_cfg(self) -> Dict[str, Any]:
        """Batches-generation parameters from this config."""
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_remainder": self.drop_remainder,
        }


@dataclasses.dataclass
class EvaluateConfig(TrainingConfig):
    """Dataclass wrapping parameters for an evaluation round."""

    drop_remainder: bool = False
