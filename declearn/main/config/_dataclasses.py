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

"""Dataclasses to wrap and parse some training-related hyperparameters."""

import dataclasses
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "EvaluateConfig",
    "FairnessConfig",
    "PrivacyConfig",
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

    The parameters wrapped by this class are those of
    `declearn.dataset.Dataset.generate_batches` and
    `declearn.communication.messaging.TrainRequest`.

    Attributes
    ----------
    batch_size: int
        Number of samples per processed data batch.
    shuffle: bool
        Whether to shuffle data samples prior to batching.
    drop_remainder: bool
        Whether to drop the last batch if it contains less
        samples than `batch_size`, or yield it anyway.
    poisson: bool
        Whether to use Poisson sampling to generate the batches.
        Useful to maintain tight Differential Privacy guarantees.
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
    poisson: bool = False
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
            "poisson": self.poisson,
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

    Exclusive attributes
    --------------------
    frequency: int
        Number of training rounds to run between evaluation ones.
        By default, run an evaluation round after each training one.

    Please refer to the parent class `TrainingConfig` for details
    on the other wrapped parameters / attributes.

    Note that `n_epoch` is dropped when this config is turned into
    an EvaluationRequest message.
    """

    frequency: int = 1
    drop_remainder: bool = False

    @property
    def message_params(self) -> Dict[str, Any]:
        """ValidRequest message parameters from this config."""
        params = super().message_params
        params.pop("n_epoch")
        return params


@dataclasses.dataclass
class PrivacyConfig:
    """Dataclass wrapping parameters to set up local differential privacy.

    The parameters wrapped by this class specify the DP-SGD algorithm [1],
    providing with a budget, an accountant method, a sensitivity clipping
    threshold, and RNG-related parameters for the noise-addition module.

    Accountants supported by Opacus 1.2.0 include:

    * rdp : Renyi-DP accountant, see [1]
    * gdp : Gaussian-DP, see [2]
    * prv : Privacy loss Random Variables privacy accountant, see [3]

    Note: for more details, refer to the Opacus source code and the
    doctrings of each accountant. See
    https://github.com/pytorch/opacus/tree/main/opacus/accountants

    Attributes
    ----------
    budget: (float, float)
        Target total privacy budget per client, expressed in terms of
        (epsilon-delta)-DP over the full training schedule.
    accountant: str
        Accounting mechanism used to estimate epsilon by Opacus.
    sclip_norm: float
        Clipping threshold of sample-wise gradients' euclidean norm.
        This parameter binds the sensitivity of sample-wise gradients.
    use_csprng: bool
        Whether to use cryptographically-secure pseudo-random numbers
        (CSPRNG) rather than the default numpy generator.
        This is significantly slower than using the default numpy RNG.
    seed: int or None
        Optional seed to the noise-addition module's RNG.
        Unused if `safe_mode=True`.

    References
    ----------
    - [1]
        Abadi et al, 2016.
        Deep Learning with Differential Privacy.
        https://arxiv.org/abs/1607.00133
    - [2]
        Dong et al, 2019.
        Gaussian Differential Privacy.
        https://arxiv.org/abs/1905.02383
    - [3]
        Gopi et al, 2021.
        Numerical Composition of Differential Privacy.
        https://arxiv.org/abs/2106.02848
    """

    budget: Tuple[float, float]
    sclip_norm: float
    accountant: str = "rdp"
    use_csprng: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        """Type- and value-check (some of) the wrapped parameters."""
        # Verify budget validity.
        if isinstance(self.budget, list):
            self.budget = tuple(self.budget)
        if not (
            isinstance(self.budget, tuple)
            and len(self.budget) == 2
            and isinstance(self.budget[0], (float, int))
            and self.budget[0] > 0
            and isinstance(self.budget[1], (float, int))
            and self.budget[1] >= 0
        ):
            raise TypeError("'budget' should be a tuple of positive floats.")
        # Verify max_norm validity.
        if not (
            isinstance(self.sclip_norm, (float, int)) and self.sclip_norm > 0
        ):
            raise TypeError("'sclip_norm' should be a positive float.")
        # Verify accountant validity.
        accountants = ("rdp", "gdp", "prv")
        if self.accountant not in accountants:
            raise TypeError(f"'accountant' should be one of {accountants}")


@dataclasses.dataclass
class FairnessConfig:
    """Dataclass wrapping parameters for fairness evaluation rounds.

    The parameters wrapped by this class are those of
    `declearn.fairness.core.FairnessAccuracyComputer`
    metrics-computation methods.

    Attributes
    ----------
    batch_size: int
        Number of samples per processed data batch.
    frequency: int
        Number of training rounds to run between fairness ones.
        By default, run a fairness round before each training one.
    n_batch: int or None, default=None
        Optional maximum number of batches to draw.
        If None, use the entire training dataset.
    thresh: float or None, default=None
        Optional binarization threshold for binary classification
        models' output scores. If None, use 0.5 by default, or 0.0
        for `SklearnSGDModel` instances.
        Unused for multinomial classifiers (argmax over scores).
    """

    batch_size: int = 32
    frequency: int = 1
    n_batch: Optional[int] = None
    thresh: Optional[float] = None
