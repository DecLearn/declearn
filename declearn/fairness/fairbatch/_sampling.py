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

"""FairBatch sampling probability controllers."""

import abc
from typing import Any, ClassVar, Dict, List, Literal, Tuple

from declearn.fairness.api import instantiate_fairness_function

__all__ = [
    "FairbatchSamplingController",
    "setup_fairbatch_controller",
]


GroupLabel = Literal["0_0", "0_1", "1_0", "1_1"]


class FairbatchSamplingController(metaclass=abc.ABCMeta):
    """ABC to compute and update Fairbatch sampling probabilities."""

    f_type: ClassVar[str]

    def __init__(
        self,
        groups: Dict[GroupLabel, Tuple[Any, ...]],
        counts: Dict[Tuple[Any, ...], int],
        alpha: float = 0.005,
        **kwargs: Any,
    ) -> None:
        """Instantiate the Fairbatch sampling probabilities controller.

        Parameters
        ----------
        groups:
            Dict mapping canonical labels to sensitive group definitions.
        counts:
            Dict mapping sensitive group definitions to sample counts.
        alpha:
            Hyper-parameter controlling the update rule for internal
            states and thereof sampling probabilities.
        **kwargs:
            Keyword arguments specific to the fairness definition in use.
        """
        # Assign input parameters as attributes.
        self.groups = groups
        self.counts = counts
        self.total = sum(counts.values())
        self.alpha = alpha
        # Initialize internal states and sampling probabilities.
        self.states = self.compute_initial_states()
        # Initialize a fairness function.
        self.f_func = instantiate_fairness_function(
            f_type=self.f_type, counts=counts, **kwargs
        )

    @abc.abstractmethod
    def compute_initial_states(
        self,
    ) -> Dict[str, float]:
        """Return a dict containing initial internal states.

        Returns
        -------
        states:
            Dict associating float values to arbitrary names that
            depend on the type of group-fairness being optimized.
        """

    @abc.abstractmethod
    def get_sampling_probas(
        self,
    ) -> Dict[Tuple[Any, ...], float]:
        """Return group-wise sampling probabilities.

        Returns
        -------
        sampling_probas:
            Dict mapping sensitive group definitions to their sampling
            probabilities, as establised via the FairBatch algorithm.
        """

    @abc.abstractmethod
    def update_from_losses(
        self,
        losses: Dict[Tuple[Any, ...], float],
    ) -> None:
        """Update internal states based on group-wise losses.

        Parameters
        ----------
        losses:
            Group-wise model loss values, as a `{group_k: loss_k}` dict.
        """

    def update_from_federated_losses(
        self,
        losses: Dict[Tuple[Any, ...], float],
    ) -> None:
        """Update internal states based on federated group-wise losses.

        Parameters
        ----------
        losses:
            Group-wise sum-aggregated local-group-count-weighted model
            loss values, computed over an ensemble of local datasets.
            I.e. `{group_k: sum_i(n_ik * loss_ik)}` dict.

        Raises
        ------
        KeyError
            If any defined sensitive group does not have a loss value.
        """
        losses = {key: val / self.counts[key] for key, val in losses.items()}
        self.update_from_losses(losses)


class FairbatchEqualityOpportunity(FairbatchSamplingController):
    """FairbatchSamplingController subclass for 'equality_of_opportunity'."""

    f_type = "equality_of_opportunity"

    def compute_initial_states(
        self,
    ) -> Dict[str, float]:
        # Gather sample counts and share with positive target label.
        nsmp_10 = self.counts[self.groups["1_0"]]
        nsmp_11 = self.counts[self.groups["1_1"]]
        p_tgt_1 = (nsmp_10 + nsmp_11) / self.total
        # Assign the initial lambda and fixed quantities to re-use.
        return {
            "lambda": nsmp_10 / self.total,
            "p_tgt_1": p_tgt_1,
            "p_g_0_0": self.counts[self.groups["0_0"]] / self.total,
            "p_g_0_1": self.counts[self.groups["0_1"]] / self.total,
        }

    def get_sampling_probas(
        self,
    ) -> Dict[Tuple[Any, ...], float]:
        return {
            self.groups["0_0"]: self.states["p_g_0_0"],
            self.groups["0_1"]: self.states["p_g_0_1"],
            self.groups["1_0"]: self.states["lambda"],
            self.groups["1_1"]: self.states["p_tgt_1"] - self.states["lambda"],
        }

    def update_from_losses(
        self,
        losses: Dict[Tuple[Any, ...], float],
    ) -> None:
        # Gather mean-aggregated losses for the two groups of interest.
        loss_10 = losses[self.groups["1_0"]]
        loss_11 = losses[self.groups["1_1"]]
        # Update lambda based on these and the alpha hyper-parameter.
        if loss_10 > loss_11:
            self.states["lambda"] = min(
                self.states["lambda"] + self.alpha, self.states["p_tgt_1"]
            )
        elif loss_10 < loss_11:
            self.states["lambda"] = max(self.states["lambda"] - self.alpha, 0)


class FairbatchEqualizedOdds(FairbatchSamplingController):
    """FairbatchSamplingController subclass for 'equalized_odds'."""

    f_type = "equalized_odds"

    def compute_initial_states(
        self,
    ) -> Dict[str, float]:
        # Gather sample counts.
        nsmp_00 = self.counts[self.groups["0_0"]]
        nsmp_01 = self.counts[self.groups["0_1"]]
        nsmp_10 = self.counts[self.groups["1_0"]]
        nsmp_11 = self.counts[self.groups["1_1"]]
        # Compute initial lambas, and attribute-wise sample counts.
        return {
            "lambda_1": nsmp_00 / self.total,
            "lambda_2": nsmp_10 / self.total,
            "p_trgt_0": (nsmp_00 + nsmp_01) / self.total,
            "p_trgt_1": (nsmp_10 + nsmp_11) / self.total,
        }

    def get_sampling_probas(
        self,
    ) -> Dict[Tuple[Any, ...], float]:
        states = self.states
        return {
            self.groups["0_0"]: states["lambda_1"],
            self.groups["0_1"]: states["p_trgt_0"] - states["lambda_1"],
            self.groups["1_0"]: states["lambda_2"],
            self.groups["1_1"]: states["p_trgt_1"] - states["lambda_2"],
        }

    def update_from_losses(
        self,
        losses: Dict[Tuple[Any, ...], float],
    ) -> None:
        # Compute loss differences for each target label.
        diff_loss_tgt_0 = (
            losses[self.groups["0_0"]] - losses[self.groups["0_1"]]
        )
        diff_loss_tgt_1 = (
            losses[self.groups["1_0"]] - losses[self.groups["1_1"]]
        )
        # Update a lambda based on these and the alpha hyper-parameter.
        if abs(diff_loss_tgt_0) > abs(diff_loss_tgt_1):
            if diff_loss_tgt_0 > 0:
                self.states["lambda_1"] = min(
                    self.states["lambda_1"] + self.alpha,
                    self.states["p_trgt_0"],
                )
            elif diff_loss_tgt_0 < 0:
                self.states["lambda_1"] = max(
                    self.states["lambda_1"] - self.alpha, 0
                )
        elif diff_loss_tgt_1 > 0:
            self.states["lambda_2"] = min(
                self.states["lambda_2"] + self.alpha,
                self.states["p_trgt_1"],
            )
        elif diff_loss_tgt_1 < 0:
            self.states["lambda_2"] = max(
                self.states["lambda_2"] - self.alpha, 0
            )


class FairbatchDemographicParity(FairbatchSamplingController):
    """FairbatchSamplingController subclass for 'demographic_parity'."""

    f_type = "demographic_parity"

    def compute_initial_states(
        self,
    ) -> Dict[str, float]:
        # Gather sample counts.
        nsmp_00 = self.counts[self.groups["0_0"]]
        nsmp_01 = self.counts[self.groups["0_1"]]
        nsmp_10 = self.counts[self.groups["1_0"]]
        nsmp_11 = self.counts[self.groups["1_1"]]
        # Compute initial lambas, and target-label-wise sample counts.
        return {
            "lambda_1": nsmp_00 / self.total,
            "lambda_2": nsmp_01 / self.total,
            "p_attr_0": (nsmp_00 + nsmp_10) / self.total,
            "p_attr_1": (nsmp_01 + nsmp_11) / self.total,
            "n_attr_0": nsmp_00 + nsmp_10,
            "n_attr_1": nsmp_01 + nsmp_11,
        }

    def get_sampling_probas(
        self,
    ) -> Dict[Tuple[Any, ...], float]:
        states = self.states
        return {
            self.groups["0_0"]: states["lambda_1"],
            self.groups["1_0"]: states["p_attr_0"] - states["lambda_1"],
            self.groups["0_1"]: states["lambda_2"],
            self.groups["1_1"]: states["p_attr_1"] - states["lambda_2"],
        }

    def update_from_losses(
        self,
        losses: Dict[Tuple[Any, ...], float],
    ) -> None:
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
        # Compute aggregated-loss differences for each target label.
        diff_loss_tgt_0 = labeled_losses["0_0"] - labeled_losses["0_1"]
        diff_loss_tgt_1 = labeled_losses["1_0"] - labeled_losses["1_1"]
        # Update a lambda based on these and the alpha hyper-parameter.
        if abs(diff_loss_tgt_0) > abs(diff_loss_tgt_1):
            if diff_loss_tgt_0 > 0:
                self.states["lambda_1"] = max(
                    self.states["lambda_1"] - self.alpha, 0
                )
            elif diff_loss_tgt_0 < 0:
                self.states["lambda_1"] = min(
                    self.states["lambda_1"] + self.alpha,
                    self.states["p_attr_0"],
                )
        elif diff_loss_tgt_1 > 0:
            self.states["lambda_2"] = min(
                self.states["lambda_2"] + self.alpha,
                self.states["p_attr_1"],
            )
        elif diff_loss_tgt_1 < 0:
            self.states["lambda_2"] = max(
                self.states["lambda_2"] - self.alpha, 0
            )


def assign_sensitive_group_labels(
    groups: List[Tuple[Any, ...]],
    target: int,
) -> Dict[GroupLabel, Tuple[Any, ...]]:
    """Parse sensitive group definitions to match canonical labels.

    Parameters
    ----------
    groups:
        List of sensitive group definitions, as a list of tuples.
        These should be four tuples arising from the intersection
        of binary labels (with any actual type).
    target:
        Value of the target label to treat as positive.

    Returns
    -------
    labeled_groups:
        Dict mapping canonical labels `"0_0", "0_1", "1_0", "1_1"`
        to the input sensitive group definitions.

    Raises
    ------
    ValueError
        If 'groups' has unproper length, values that do not appear
        to be binary, or that do not match the specified 'target'.
    """
    # Verify that groups can be identified as crossing two binary labels.
    if len(groups) != 4:
        raise ValueError(
            "FairBatch requires counts over exactly 4 sensitive groups, "
            "arising from a binary target label and a binary sensitive "
            "attribute."
        )
    target_values = list({group[0] for group in groups})
    s_attr_values = sorted(list({group[1] for group in groups}))
    if not len(target_values) == len(s_attr_values) == 2:
        raise ValueError(
            "FairBatch requires sensitive groups to arise from a binary "
            "target label and a binary sensitive attribute."
        )
    # Identify the positive and negative label values.
    if target_values[0] == target:
        postgt, negtgt = target_values
    elif target_values[1] == target:
        negtgt, postgt = target_values
    else:
        raise ValueError(
            f"Received a target value of '{target}' that does not match any "
            f"value in the sensitive group definitions: {target_values}."
        )
    # Match group definitions with canonical string labels.
    return {
        "0_0": (negtgt, s_attr_values[0]),
        "0_1": (negtgt, s_attr_values[1]),
        "1_0": (postgt, s_attr_values[0]),
        "1_1": (postgt, s_attr_values[1]),
    }


def setup_fairbatch_controller(
    f_type: str,
    counts: Dict[Tuple[Any, ...], int],
    target: int = 1,
    alpha: float = 0.005,
) -> FairbatchSamplingController:
    """Instantiate a FairBatch sampling probabilities controller.

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
    """
    controller_types = {
        "demographic_parity": FairbatchDemographicParity,
        "equality_of_opportunity": FairbatchEqualityOpportunity,
        "equalized_odds": FairbatchEqualizedOdds,
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
