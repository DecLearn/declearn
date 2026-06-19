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

"""TrainingManager subclass implementing Differential Privacy mechanisms."""

import logging
from typing import List, Optional, Tuple, Union

from opacus.accountants import IAccountant, create_accountant  # type: ignore
from opacus.accountants.utils import get_noise_multiplier  # type: ignore

from declearn import messaging
from declearn.aggregator import Aggregator
from declearn.dataset import Dataset
from declearn.metrics import MetricInputType, MetricSet
from declearn.model.api import Model
from declearn.optimizer import Optimizer
from declearn.optimizer.modules import GaussianNoiseModule
from declearn.training import TrainingManager
from declearn.typing import Batch

__all__ = [
    "DPTrainingManager",
]


class DPTrainingManager(TrainingManager):
    """TrainingManager subclass adding Differential Privacy mechanisms.

    This class extends the base TrainingManager class in three key ways:

    * Perform per-sample gradients clipping (through the Model API),
      parametrized by the added, optional `sclip_norm` attribute.
    * Add noise to batch-averaged gradients at each step of training,
      calibrated from an (epsilon, delta) DP budget and the planned
      training computational effort (number of steps, sample rate...).
    * Keep track of the spent privacy budget during training, and block
      training once the monitored budget is fully spent (early-stop the
      training routine if the next step would result in over-spending).

    This TrainingManager therefore implements the differentially-private
    stochastic gradient descent algorithm (DP-SGD) [1] algorithm, in a
    modular fashion that enables using any kind of optimizer plug-in
    supported by its (non-DP) parent.

    References
    ----------
    [1] Abadi et al, 2016.
        Deep Learning with Differential Privacy.
        https://arxiv.org/abs/1607.00133
    """

    # pylint: disable-next=too-many-positional-arguments
    def __init__(  # noqa: PLR0913
        self,
        model: Model,
        optim: Optimizer,
        aggrg: Aggregator,
        train_data: Dataset,
        valid_data: Optional[Dataset] = None,
        metrics: Union[MetricSet, List[MetricInputType], None] = None,
        logger: Union[logging.Logger, str, None] = None,
        verbose: bool = True,
    ) -> None:
        # inherited signature; pylint: disable=too-many-arguments
        super().__init__(
            model=model,
            optim=optim,
            aggrg=aggrg,
            train_data=train_data,
            valid_data=valid_data,
            metrics=metrics,
            logger=logger,
            verbose=verbose,
        )
        # Add DP-related fields: accountant, clipping norm and budget.
        self.accountant: Optional[IAccountant] = None
        self.sclip_norm: Optional[float] = None
        self._dp_budget = (0.0, 0.0)
        self._dp_states: Optional[Tuple[float, float]] = None
        # Precompute the max-allowed number of steps once per round via a
        # binary search over the privacy accountant, collapsing the per-step
        # budget check to a single integer compare. Mid-round budget
        # detection is preserved exactly (see `_compute_max_steps_for_round`).
        self._max_steps_this_round: Optional[int] = None
        self._step_counter_this_round: int = 0

    def make_private(
        self,
        message: messaging.PrivacyRequest,
    ) -> None:
        """Set up the use of DP-SGD based on a received PrivacyRequest.

        Parameters
        ----------
        message: PrivacyRequest
            PrivacyRequest message specifying the privacy budget, type
            of accountant and expected use of the training data.
        """
        # REVISE: add support for fixed requested noise multiplier
        # Compute the noise multiplier to use based on the budget
        # and the planned training duration and parameters.
        noise_multiplier = self._fit_noise_multiplier(
            budget=message.budget,
            n_samples=self.train_data.get_data_specs().n_samples,
            batch_size=message.batches["batch_size"],
            n_round=message.rounds,
            n_epoch=message.n_epoch,
            n_steps=message.n_steps,
            accountant=message.accountant,
            drop_remainder=message.batches.get("drop_remainder", True),
        )
        # Add a gaussian noise addition module to the optimizer's pipeline.
        noise_module = GaussianNoiseModule(
            std=noise_multiplier * message.sclip_norm,
            safe_mode=message.use_csprng,
            seed=message.seed,
        )
        self.optim.modules.insert(0, noise_module)
        # Create an accountant and store the clipping norm and privacy budget.
        self.accountant = create_accountant(message.accountant)
        self.sclip_norm = message.sclip_norm
        self._dp_budget = message.budget

    # pylint: disable=too-many-positional-arguments
    def _fit_noise_multiplier(  # noqa: PLR0913
        self,
        budget: Tuple[float, float],
        n_samples: int,
        batch_size: int,
        n_round: int,
        n_epoch: Optional[int] = None,
        n_steps: Optional[int] = None,
        accountant: str = "rdp",
        drop_remainder: bool = True,
    ) -> float:
        """Parametrize a DP noise multiplier based on a training schedule."""
        # arguments are all required; pylint: disable=too-many-arguments
        # Compute the expected number of batches per epoch.
        n_batches = n_samples // batch_size
        if not drop_remainder:
            n_batches += bool(n_batches % batch_size)
        # Compute the total number of steps that will be performed.
        steps = n_round
        if n_epoch and n_steps:
            steps *= min(n_steps, n_epoch * n_batches)
        elif n_steps:  # i.e. n_epoch is None
            steps *= n_steps
        elif n_epoch:  # i.e. n_steps is None
            steps *= n_epoch * n_batches
        else:  # i.e. both None: then default n_epoch=1 is used
            steps *= n_batches
            if n_epoch is None:
                self.logger.warning(
                    "Both `n_epoch` and `n_steps` are None in the received "
                    "PrivacyRequest. As a result, the noise used for DP is "
                    "calibrated assuming `n_epoch=1` per round, which might "
                    "be wrong, and result in under- or over-spending the "
                    "privacy budget during the actual training rounds."
                )
        # Use the former information to choose the noise multiplier.
        return get_noise_multiplier(
            target_epsilon=budget[0],
            target_delta=budget[1],
            sample_rate=batch_size / n_samples,
            steps=steps,
            accountant=accountant,
        )

    def get_noise_multiplier(self) -> Optional[float]:
        """Return the noise multiplier used for DP-SGD, if any.

        Returns
        -------
        noise_multiplier: float or None
            Standard deviation of the gaussian noise-addition module
            placed at the start of the wrapped optimizer's pipeline,
            if one is indeed present.
        """
        if self.optim.modules:
            if isinstance(self.optim.modules[0], GaussianNoiseModule):
                return self.optim.modules[0].std / (self.sclip_norm or 1.0)
        return None

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Return the (epsilon, delta) privacy budget spent so far.

        Returns
        -------
        epsilon: float
            epsilon component of the privacy budget spent.
        delta: float
            delta component of the privacy budget spent.
        """
        if self.accountant is None:
            raise RuntimeError("Cannot return spent privacy: DP is not used.")
        delta = self._dp_budget[1]
        epsilon = self.accountant.get_epsilon(delta=delta)
        return epsilon, delta

    def run_train_step(
        self,
        batch: Batch,
    ) -> None:
        # Optionally have the DP accountant authorize or prevent the step.
        # Note that once the step is authorized, it is also accounted for.
        self._prevent_budget_overspending()
        # Use fixed-threshold sample-wise gradients clipping, in addition
        # to all the features implemented at the parent level.
        # Note: in the absence of `make_private`, no clipping is performed.
        self.optim.run_train_step(self.model, batch, sclip=self.sclip_norm)

    def _prevent_budget_overspending(self) -> None:
        """Raise a StopIteration if a step would overspend the DP budget.

        This method relies on the private attributes `_dp_states` and
        `_max_steps_this_round` to have been properly set as part of the
        `_training_round` routine.

        The maximum number of steps allowed this round was precomputed
        once at round start via a binary search over the privacy
        accountant (see `_compute_max_steps_for_round`). The per-step cost
        is therefore a single (cheap) `accountant.step` plus an integer
        compare. Mid-round budget detection is preserved exactly, since the
        precomputed bound is derived from the same ε computation the
        canonical per-step check would have performed.
        """
        if self.accountant is not None and self._dp_states is not None:
            noise, srate = self._dp_states
            self.accountant.step(noise_multiplier=noise, sample_rate=srate)
            self._step_counter_this_round += 1
            if (
                self._max_steps_this_round is not None
                and self._step_counter_this_round > self._max_steps_this_round
            ):
                # Remove the step from the history as it will not be taken.
                last = self.accountant.history.pop(-1)
                if last[-1] > 1:  # number of steps with that (noise, srate)
                    last = (last[0], last[1], last[2] - 1)
                    self.accountant.history.append(last)
                # Prevent the step from being taken.
                raise StopIteration(
                    "Local DP budget would be exceeded by taking the next "
                    "training step."
                )

    def _compute_max_steps_for_round(
        self, noise: float, srate: float, max_probe: int = 100_000
    ) -> int:
        """Return the largest number of steps this round keeps ε at-or-below.

        Binary-search the largest ``k`` such that adding ``k`` steps with the
        given ``(noise, srate)`` keeps the total spent epsilon at or below the
        budget.

        Implementation note: every opacus accountant (``rdp``, ``gdp`` and
        ``prv``) stores its history as ``(noise, srate, count)`` tuples, and
        computes epsilon from those aggregate counts rather than from
        individual steps. A single tuple with ``count=k`` is therefore
        equivalent to calling ``step`` ``k`` times (privacy loss depends only
        on the aggregate counts, not on step granularity or ordering). The
        probe history is built by merging into the trailing tuple (rather than
        appending a separate one) so this equivalence also holds for the
        ``gdp`` accountant, whose ``get_epsilon`` reads only the last history
        tuple -- see the body below. Each binary-search probe is thus one
        ``get_epsilon`` call on a history of size O(rounds), regardless of the
        per-round step count. With ~log2(max_probe) probes per round, the
        precompute cost is bounded and independent of the number of steps
        actually taken.

        Parameters
        ----------
        noise:
            Noise multiplier used for the upcoming round's steps.
        srate:
            Sample rate used for the upcoming round's steps.
        max_probe:
            Upper bound for the binary search (maximum number of steps it
            will ever authorize in a single round).

        Returns
        -------
        max_steps:
            Largest number of additional steps that keeps total ε ≤ budget.
        """
        if self.accountant is None:
            return 0
        budget_eps = self._dp_budget[0]
        budget_delta = self._dp_budget[1]
        # Snapshot the real history and restore it at the end, so that the
        # accountant reflects only the steps that have ACTUALLY occurred.
        history_snapshot = list(self.accountant.history)
        # Build the probe history exactly as `candidate` real `step` calls
        # would: merge into the trailing tuple when it shares this round's
        # (noise, srate), otherwise start a fresh tuple. This mirrors
        # opacus's `IAccountant.step` merge logic. It is required for the
        # GaussianAccountant ("gdp"), whose `get_epsilon` only reads the
        # LAST history tuple: appending a separate tuple would make it
        # ignore the budget already spent in previous rounds and thus
        # over-authorize steps. RDP and PRV compose additively over the
        # whole history, so merging is equivalent for them as well.
        if history_snapshot and history_snapshot[-1][:2] == (noise, srate):
            base = history_snapshot[:-1]
            prev_count = history_snapshot[-1][2]
        else:
            base = history_snapshot
            prev_count = 0
        try:
            max_feasible, search_upper = 0, max_probe
            while max_feasible < search_upper:
                candidate = (max_feasible + search_upper + 1) // 2
                self.accountant.history = base + [
                    (noise, srate, prev_count + candidate)
                ]
                eps = self.accountant.get_epsilon(delta=budget_delta)
                if eps <= budget_eps:
                    max_feasible = candidate
                else:
                    search_upper = candidate - 1
            return max_feasible
        finally:
            self.accountant.history = history_snapshot

    def _training_round(
        self,
        message: messaging.TrainRequest,
    ) -> messaging.TrainReply:
        # When using differential privacy, store accountant-required values.
        if self.accountant is not None:
            n_smp = self.train_data.get_data_specs().n_samples
            srate: float = message.batches["batch_size"] / n_smp
            noise = self.get_noise_multiplier()
            if noise is None:
                raise RuntimeError(
                    "Noise multiplier not found: something is wrong with "
                    "the local DP setup."
                )
            self._dp_states = (noise, srate)
            # Precompute the max-allowed number of steps once, and reset
            # the per-round step counter.
            self._max_steps_this_round = self._compute_max_steps_for_round(
                noise, srate
            )
            self._step_counter_this_round = 0
        # Delegate all of the actual training routine to the parent class.
        # DP budget saturation will cause training to be interrupted.
        reply = super()._training_round(message)
        # When using DP, clean up things and log about the spent budget.
        if self.accountant is not None:
            self._dp_states = None  # remove now out-of-scope state values
            self._max_steps_this_round = None
            self.logger.info(
                "Local DP budget spent at the end of the round: %s",
                self.get_privacy_spent(),
            )
        return reply
