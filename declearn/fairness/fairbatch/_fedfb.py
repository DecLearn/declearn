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

"""FedFB sampling probability controllers."""

from typing import Any, Dict, Tuple

import numpy as np

from declearn.fairness.fairbatch._sampling import (
    FairbatchDemographicParity,
    FairbatchEqualityOpportunity,
    FairbatchEqualizedOdds,
    FairbatchSamplingController,
    assign_sensitive_group_labels,
)

__all__ = [
    "setup_fedfb_controller",
]


class FedFBEqualityOpportunity(FairbatchEqualityOpportunity):
    """FedFB variant of Equality of Opportunity controller.

    This variant introduces two changes as compared with our FedFairBatch:
    - The lambda parameter and difference of losses are written with a
      different group ordering, albeit resulting in identical results.
    - When comparing loss values over sensitive groups, the notations from
      the FedFB paper indicate that the sums of losses over samples in the
      groups are compared, rather than the averages of group-wise losses;
      this implementation sticks to the FedFB paper.
    """

    f_type = "equality_of_opportunity"

    def get_sampling_probas(
        self,
    ) -> Dict[Tuple[Any, ...], float]:
        # Revert the sense of lambda (invert (1, 0) and (0, 1) groups)
        # to stick with notations from the FedFB paper.
        probas = super().get_sampling_probas()
        label_10 = self.groups["1_0"]
        label_11 = self.groups["1_1"]
        probas[label_10], probas[label_11] = probas[label_11], probas[label_10]
        return probas

    def update_from_losses(
        self,
        losses: Dict[Tuple[Any, ...], float],
    ) -> None:
        # Recover sum-aggregated losses for the two groups of interest.
        # Do not scale: obtain sums of sample losses for each group.
        # This differs from parent class (and centralized FairBatch)
        # but sticks with the FedFB paper's notations and algorithm.
        loss_10 = losses[self.groups["1_0"]] * self.counts[self.groups["1_0"]]
        loss_11 = losses[self.groups["1_1"]] * self.counts[self.groups["1_1"]]
        # Update lambda based on these and the alpha hyper-parameter.
        # Note: this is the same as in parent class, inverting sense of
        # groups (0, 0) and (1, 0), to stick with the FedFB paper.
        if loss_11 > loss_10:
            self.states["lambda"] = min(
                self.states["lambda"] + self.alpha, self.states["p_tgt_1"]
            )
        elif loss_11 < loss_10:
            self.states["lambda"] = max(self.states["lambda"] - self.alpha, 0)


class FedFBEqualizedOdds(FairbatchEqualizedOdds):
    """FedFB variant of Equalized Odds controller.

    This variant introduces three changes as compared with our FedFairBatch:
    - The lambda parameters and difference of losses are written with a
      different group ordering, albeit resulting in identical results.
    - When comparing loss values over sensitive groups, the notations from
      the FedFB paper indicate that the sums of losses over samples in the
      groups are compared, rather than the averages of group-wise losses;
      this implementation sticks to the FedFB paper.
    - The update rule for lambda parameters has a distinct formula, with the
      alpha hyper-parameter being here scaled by the difference in losses
      and normalized by the L2 norm of differences in losses, and both groups'
      lambda being updated at each step.
    """

    f_type = "equalized_odds"

    def compute_initial_states(
        self,
    ) -> Dict[str, float]:
        # Switch lambdas: apply to groups (-, 1) rather than (-, 0).
        states = super().compute_initial_states()
        states["lambda_1"] = states["p_trgt_0"] - states["lambda_1"]
        states["lambda_2"] = states["p_trgt_1"] - states["lambda_2"]
        return states

    def get_sampling_probas(
        self,
    ) -> Dict[Tuple[Any, ...], float]:
        # Rewrite the rules entirely, effectively swapping (0,1)/(0,0)
        # and (1,1)/(1,0) groups compared with parent implementation.
        states = self.states
        return {
            self.groups["0_0"]: states["p_trgt_0"] - states["lambda_1"],
            self.groups["0_1"]: states["lambda_1"],
            self.groups["1_0"]: states["p_trgt_1"] - states["lambda_2"],
            self.groups["1_1"]: states["lambda_2"],
        }

    def update_from_losses(
        self,
        losses: Dict[Tuple[Any, ...], float],
    ) -> None:
        # Recover sum-aggregated losses for each sensitive group.
        # Do not scale: obtain sums of sample losses for each group.
        # This differs from parent class (and centralized FairBatch)
        # but sticks with the FedFB paper's notations and algorithm.
        labeled_losses = {
            label: losses[group] * self.counts[group]
            for label, group in self.groups.items()
        }
        # Compute aggregated-loss differences for each target label.
        diff_loss_tgt_0 = labeled_losses["0_1"] - labeled_losses["0_0"]
        diff_loss_tgt_1 = labeled_losses["1_1"] - labeled_losses["1_0"]
        # Compute the euclidean norm of these values.
        den = float(np.linalg.norm([diff_loss_tgt_0, diff_loss_tgt_1], ord=2))
        # Update lambda_1 (affecting groups with y=0).
        update = self.alpha * diff_loss_tgt_0 / den
        self.states["lambda_1"] = min(
            self.states["lambda_1"] + update, self.states["p_trgt_0"]
        )
        self.states["lambda_1"] = max(self.states["lambda_1"], 0)
        # Update lambda_1 (affecting groups with y=1).
        update = self.alpha * diff_loss_tgt_1 / den
        self.states["lambda_2"] = min(
            self.states["lambda_2"] + update, self.states["p_trgt_1"]
        )
        self.states["lambda_2"] = max(self.states["lambda_2"], 0)


class FedFBDemographicParity(FairbatchDemographicParity):
    """FairbatchSamplingController subclass for 'demographic_parity'."""

    f_type = "demographic_parity"

    def update_from_losses(
        self,
        losses: Dict[Tuple[Any, ...], float],
    ) -> None:
        # NOTE: losses' aggregation does not defer from parent class.
        # pylint: disable=duplicate-code
        # Recover sum-aggregated losses for each sensitive group.
        # Obtain {k: n_k * Sum(loss for all samples in group k)}.
        labeled_losses = {
            label: losses[group] * self.counts[group]
            for label, group in self.groups.items()
        }
        # Normalize losses based on sensitive attribute counts.
        # Obtain {k: sum(loss for samples in k) / n_samples_with_attr}.
        labeled_losses["0_0"] /= self.states["n_attr_0"]
        labeled_losses["0_1"] /= self.states["n_attr_1"]
        labeled_losses["1_0"] /= self.states["n_attr_0"]
        labeled_losses["1_1"] /= self.states["n_attr_1"]
        # NOTE: this is where things differ from parent class.
        # pylint: enable=duplicate-code
        # Compute an overall fairness value based on all losses.
        f_val = (
            -labeled_losses["0_0"]
            + labeled_losses["0_1"]
            + labeled_losses["1_0"]
            - labeled_losses["1_1"]
            + self.counts[self.groups["0_0"]] / self.states["n_attr_0"]
            - self.counts[self.groups["0_1"]] / self.states["n_attr_1"]
        )
        # Update both lambdas based on this overall value.
        # Note: in the binary attribute case, $mu_a / ||mu||_2$
        # is equal to $sign(mu_1) / sqrt(2)$.
        update = float(np.sign(f_val) * self.alpha / np.sqrt(2))
        self.states["lambda_1"] = min(
            self.states["lambda_1"] - update, self.states["p_attr_0"]
        )
        self.states["lambda_1"] = max(self.states["lambda_1"], 0)
        self.states["lambda_2"] = min(
            self.states["lambda_2"] - update, self.states["p_attr_1"]
        )
        self.states["lambda_2"] = max(self.states["lambda_2"], 0)


def setup_fedfb_controller(
    f_type: str,
    counts: Dict[Tuple[Any, ...], int],
    target: int = 1,
    alpha: float = 0.005,
) -> FairbatchSamplingController:
    """Instantiate a FedFB sampling probabilities controller.

    This is a drop-in replacement for `setup_fairbatch_controller`
    that implemented update rules matching the Fed-FB algorithm(s)
    as introduced in [1].

    Parameters
    ----------
    f_type:
        Type of group fairness to optimize for.
    counts:
        Dict mapping sensitive group definitions to their total
        sample counts (across clients). These groups must arise
        from the crossing of a binary target label and a binary
        sensitive attribute.
    target:
        Target label to treat as positive.
    alpha:
        Alpha hyper-parameter, scaling the magnitude of sampling
        probabilities' updates by the returned controller.

    Returns
    -------
    controller:
        FairBatch sampling probabilities controller matching inputs.

    Raises
    ------
    KeyError
        If `f_type` does not match any known or supported fairness type.
    ValueError
        If `counts` keys cannot be matched to canonical group labels.

    References
    ----------
    [1] Zeng et al. (2022).
        Improving Fairness via Federated Learning.
        https://arxiv.org/abs/2110.15545
    """
    # known duplicate of fairbatch setup; pylint: disable=duplicate-code
    controller_types = {
        "demographic_parity": FedFBDemographicParity,
        "equality_of_opportunity": FedFBEqualityOpportunity,
        "equalized_odds": FedFBEqualizedOdds,
    }
    controller_cls = controller_types.get(f_type, None)
    if controller_cls is None:
        raise KeyError(
            "Unknown or unsupported fairness type parameter for FairBatch "
            f"controller initialization: '{f_type}'. Supported values are "
            f"{list(controller_types)}."
        )
    # Match groups to canonical labels and instantiate the controller.
    groups = assign_sensitive_group_labels(groups=list(counts), target=target)
    kwargs = {"target": target} if f_type == "equality_of_opportunity" else {}
    return controller_cls(  # type: ignore[abstract]
        groups=groups, counts=counts, alpha=alpha, **kwargs
    )
