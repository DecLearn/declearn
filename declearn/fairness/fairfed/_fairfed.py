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

"""Fairfed-specific synthetic fairness value computer."""

import warnings
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "FairfedValueComputer",
]


class FairfedValueComputer:
    """Fairfed-specific synthetic fairness value computer.

    Strict mode
    -----------
    This FairFed implementation comes in two flavors.

    - The "strict" mode sticks to the original FairFed paper:
        - It applies only to binary classification tasks with
            a single binary sensitive attributes.
        - Clients must hold examples to each and every group.
        - If `f_type` is not explicitly cited in the original
            paper, a `RuntimeWarning` is warned.
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

    def __init__(
        self,
        f_type: str,
        strict: bool = True,
        target: int = 1,
    ) -> None:
        """Instantiate the FairFed-specific fairness function wrapper.

        Parameters
        ----------
        f_type:
            Name of the fairness definition being optimized.
        strict:
            Whether to stick strictly to the FairFed paper's setting
            and explicit formulas, or to use a broader adaptation of
            FairFed to more diverse settings.
            See class docstring for details.
        target:
            Choice of target label to focus on in `strict` mode.
            Unused when `strict=False`.
        """
        self.f_type = f_type
        self.strict = strict
        self.target = target
        self._key_groups: Optional[Tuple[Tuple[Any, ...], Tuple[Any, ...]]] = (
            None
        )

    def initialize(
        self,
        groups: List[Tuple[Any, ...]],
    ) -> None:
        """Initialize the Fairfed synthetic value computer from group counts.

        Parameters
        ----------
        groups:
            List of sensitive group definitions.
        """
        if self.strict:
            self._key_groups = self.identify_key_groups(groups)

    def identify_key_groups(
        self,
        groups: List[Tuple[Any, ...]],
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
                stacklevel=2,
            )
        if len(groups) != 4:
            raise RuntimeError(
                "FairFed in 'strict' mode requires exactly 4 sensitive groups,"
                " arising from a binary target label and a binary attribute."
            )
        key_groups = tuple(sorted([g for g in groups if g[0] == self.target]))
        if len(key_groups) != 2:
            raise KeyError(
                f"Failed to identify the (target,attr_0);(target,attr_1) "
                "pair of sensitive groups for FairFed in 'strict' mode "
                f"with 'target' value {self.target}."
            )
        return key_groups

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
