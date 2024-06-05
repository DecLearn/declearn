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

"""FairFed-specific fairness function wrapper."""

import warnings
from typing import Any, Dict, Optional, Tuple


from declearn.fairness.api import FairnessFunction


class FairfedFairnessFunction:
    """FairFed-specific fairness function wrapper."""

    def __init__(
        self,
        wrapped: FairnessFunction,
        strict: bool = True,
        target: Optional[int] = None,
    ) -> None:
        """Instantiate the FairFed-specific fairness function wrapper.

        Parameters
        ----------
        wrapped:
            Initial `FairnessFunction` instance to wrap up for FairFed.
        strict:
            Whether to stick strictly to the FairFed paper's setting
            and explicit formulas, or to use a broader adaptation of
            FairFed to more diverse settings.
            See details below.
        target:
            Optional choice of target label to focus on in `strict` mode.
            Only used when `strict=True`. If `None`, use `wrapped.target`
            when it exists, or else a default value of 1.

        Strict mode
        -----------
        This FairFed implementation comes in two flavors.

        - The "strict" mode sticks to the original FairFed paper:
            - It applies only to binary classification tasks with
              a single binary sensitive attributes.
            - Clients must hold examples to each and every group.
            - If `wrapped.f_type` is not explicitly cited in the
              original paper, a `RuntimeWarning` is warned.
            - The synthetic fairness value is computed based on
              fairness values for two groups: (y=`target`,s=1)
              and (y=`target`,s=0).

        - The "non-strict" mode extends to broader settings:
            - It applies to any number of sensitive groups.
            - Clients may not hold examples of all groups.
            - It applies to any type of group-fairness.
            - The synthetic fairness value is computed as
              the average of all absolute fairness values.
            - The local fairness is only computed over groups
              that have a least one sample in the local data.
        """
        self.wrapped = wrapped
        self._key_groups = (
            None
        )  # type: Optional[Tuple[Tuple[Any, ...], Tuple[Any, ...]]]
        if strict:
            target = int(
                getattr(wrapped, "target", 1) if target is None else target
            )
            self._key_groups = self._identify_key_groups(target)

    @property
    def f_type(
        self,
    ) -> str:
        """Type of group-fairness being measured."""
        return self.wrapped.f_type

    @property
    def strict(
        self,
    ) -> bool:
        """Whether this function strictly sticks to the FairFed paper."""
        return self._key_groups is not None

    def _identify_key_groups(
        self,
        target: int,
    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Parse sensitive groups' definitions to identify 'key' ones."""
        if self.f_type not in (
            "demographic_parity",
            "equality_of_opportunity",
            "equalized_odds",
        ):
            warnings.warn(
                f"Using fairness type '{self.f_type}' with FairFed in 'strict'"
                " mode. This is supported, but beyond the original paper.",
                RuntimeWarning,
            )
        if len(self.wrapped.groups) != 4:
            raise RuntimeError(
                "FairFed in 'strict' mode requires exactly 4 sensitive groups,"
                " arising from a binary target label and a binary attribute."
            )
        groups = tuple(
            sorted([grp for grp in self.wrapped.groups if grp[0] == target])
        )
        if len(groups) != 2:
            raise KeyError(
                f"Failed to identify the (target,attr_0);(target,attr_1) "
                "pair of sensitive groups for FairFed in 'strict' mode "
                f"with 'target' value {target}."
            )
        return groups

    def compute_group_fairness_from_accuracy(
        self,
        accuracy: Dict[Tuple[Any, ...], float],
        federated: bool,
    ) -> Dict[Tuple[Any, ...], float]:
        """Compute group-wise fairness values from group-wise accuracy metrics.

        Parameters
        ----------
        accuracy:
            Group-wise accuracy values of the model being evaluated on a
            dataset. I.e. `{group_k: P(y_pred == y_true | group_k)}`.
        federated:
            Whether `accuracy` holds values computes federatively, that
            is sum-aggregated local-group-count-weighted accuracies
            `{group_k: sum_i(n_ik * accuracy_ik)}`.

        Returns
        -------
        fairness:
            Group-wise fairness metrics, as a `{group_k: score_k}` dict.
        """
        if federated:
            return self.wrapped.compute_from_federated_group_accuracy(accuracy)
        return self.wrapped.compute_from_group_accuracy(accuracy)

    def compute_synthetic_fairness_value(
        self,
        fairness: Dict[Tuple[Any, ...], float],
    ) -> float:
        """Compute a synthetic fairness value from group-wise ones.

        If `self.strict`, compute the difference between the fairness
        values associated with two key sensitive groups, as per the
        original FairFed paper for the two definitions exposed by the
        authors.

        Otherwise, compute the average of absolute group-wise fairness
        values, that applies to more generic fairness formulations than
        in the original paper, and may encompass broader information.

        Parameters
        ----------
        fairness:
            Group-wise fairness metrics, as a `{group_k: score_k}` dict.

        Returns
        -------
        value:
            Scalar value summarizing the computed fairness.
        """
        if self._key_groups is None:
            return sum(abs(x) for x in fairness.values()) / len(fairness)
        return fairness[self._key_groups[0]] - fairness[self._key_groups[1]]
