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

"""Concrete implementations of various group-fairness functions."""

from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np

from declearn.fairness.api import FairnessFunction
from declearn.utils import access_types_mapping

__all__ = (
    "AccuracyParityFunction",
    "DemographicParityFunction",
    "EqualityOfOpportunityFunction",
    "EqualizedOddsFunction",
    "list_fairness_functions",
)


def list_fairness_functions() -> Dict[str, Type[FairnessFunction]]:
    """Return a mapping of registered FairnessFunction subclasses.

    This function aims at making it easy for end-users to list and access
    all available FairnessFunction classes at any given time. The returned
    dict uses unique identifier keys, which may be used to use the associated
    function within a [declearn.fairness.api.FairnessControllerServer][].

    Note that the mapping will include all declearn-provided functions,
    but also registered functions provided by user or third-party code.

    See also
    --------
    * [declearn.fairness.api.FairnessFunction][]:
        API-defining abstract base class for the FairnessFunction classes.

    Returns
    -------
    mapping:
        Dictionary mapping unique str identifiers to `FairnessFunction`
        class constructors.
    """
    return access_types_mapping("FairnessFunction")


class AccuracyParityFunction(FairnessFunction):
    """Accuracy Parity group-fairness function.

    Definition
    ----------

    Accuracy Parity is achieved when

    $$ \\forall r \\in S , P(h(x) = y | s = r) == P(h(x) = y) $$

    where $S$ denotes possible values of a (set of intersected) sensitive
    attribute(s), $y$ is the true target classification label and $h$ is
    the evaluated classifier.

    In other words, Accuracy Parity is achieved when the model's accuracy
    is independent from the sensitive attribute(s) (but not necessarily
    balanced across specific target classes).

    Formula
    -------

    For any sensitive group $k = (l, r)$ defined by the intersection of a
    given true label and a value of the sensitive attribute(s), Accuracy
    Parity can be expressed in the canonical form from [1]:

    $$ F_k(h, T) = C_k^0 + \\sum_{k'} C_k^{k'} P(h(x) \\neq y | T_{k'}) $$

    using the following constants:

    - $ C_k^{k'} = (n_{k'} / n) - 1{k_s = k'_s} * (n_{k'} / n_s) $
    - $ C_k^0 = 0 $

    where $n$ denotes a number of samples in the empirical dataset used,
    and its subscripted counterparts are number of samples that belong to
    a given sensitive group and/or have given sensitive attribute(s) value.

    This results in all scores associated with a given target label to all
    be equal, as the partition in sensitive groups could be done regardless
    of the target label.

    References
    ----------
    [1] Maheshwari & Perrot (2023).
        FairGrad: Fairness Aware Gradient Descent.
        https://openreview.net/forum?id=0f8tU3QwWD
    """

    f_type = "accuracy_parity"

    def compute_fairness_constants(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        counts = self._counts
        # Compute the sensitive-attributes-wise and total number of samples.
        s_tot: Dict[Any, int] = {}  # attributes-wise number of samples
        for categ, count in counts.items():
            s_tot[categ[1:]] = s_tot.get(categ[1:], 0) + count
        total = sum(s_tot.values())
        # Compute fairness constraint constants C_k^k'.
        c_kk = np.zeros((len(counts), len(counts)))
        for idx, cat_i in enumerate(counts):
            for jdx, cat_j in enumerate(counts):
                coef = counts[cat_j] / total
                if cat_i[1:] == cat_j[1:]:  # same sensitive attributes
                    coef -= counts[cat_j] / s_tot[cat_i[1:]]
                c_kk[idx, jdx] = coef
        # Return the computed constants.
        c_k0 = np.array([0.0])
        return c_k0, c_kk


class DemographicParityFunction(FairnessFunction):
    """Demographic Parity group-fairness function for binary classifiers.

    Note that this implementation is restricted to binary classification.

    Definition
    ----------

    Demographic Parity is achieved when

    $$ \\forall l \\in Y, \\forall r \\in S,
       P(h(x) = l | s = r) == P(h(x) = l) $$

    where $Y$ denotes possible target labels, $S$ denotes possible values
    of a (set of intersected) sensitive attribute(s), and $h$ is the
    evaluated classifier.

    In other words, Demographic Parity is achieved when the probability to
    predict any given label is independent from the sensitive attribute(s)
    (regardless of whether that label is accurate or not).

    Formula
    -------

    When considering a binary classification task, for any sensitive group
    $k = (l, r)$ defined by the intersection of a given true label and a
    value of the sensitive attribute(s), Demographic Parity can be expressed
    in the canonical form from [1]:

    $$ F_k(h, T) = C_k^0 + \\sum_{k'} C_k^{k'} P(h(x) \\neq y | T_{k'}) $$

    using the following constants:

    - $ C_{l,r}^0 = (n_k / n_r) - (n_l / n) $
    - $ C_{l,r}^{l,r} = (n_k / n) - (n_k / n_r) $
    - $ C_{l,r}^{l',r} = (n_{k'} / n_r) - (n_{k'} / n)  $
    - $ C_{l,r}^{l,r'} = n_{k'} / n $
    - $ C_{l,r}^{l',r'} =  - n_{k'} / n $

    where $n$ denotes a number of samples in the empirical dataset used,
    and its subscripted counterparts are number of samples that belong to
    a given sensitive group and/or have given  sensitive attribute(s) or
    true label values.

    References
    ----------
    [1] Maheshwari & Perrot (2023).
        FairGrad: Fairness Aware Gradient Descent.
        https://openreview.net/forum?id=0f8tU3QwWD
    """

    f_type = "demographic_parity"

    def compute_fairness_constants(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        counts = self._counts
        # Check that the labels are binary.
        if len(set(key[0] for key in counts)) != 2:
            raise ValueError("Demographic Parity requires binary labels.")
        # Compute label-wise, attribute-wise and total number of samples.
        l_tot: Dict[Any, int] = {}  # label-wise number of samples
        s_tot: Dict[Tuple[Any, ...], int] = {}  # attribute-wise
        for categ, count in counts.items():
            l_tot[categ[0]] = l_tot.get(categ[0], 0) + count
            s_tot[categ[1:]] = s_tot.get(categ[1:], 0) + count
        total = sum(l_tot.values())
        # Compute fairness constraint constants C_k^k'.
        c_k0 = np.zeros(len(counts))
        c_kk = np.zeros((len(counts), len(counts)))
        for idx, cat_i in enumerate(counts):
            # Compute the C_k^0 constant.
            c_k0[idx] = counts[cat_i] / s_tot[cat_i[1:]]
            c_k0[idx] -= l_tot[cat_i[0]] / total
            # Compute all other C_k^k' constants.
            for jdx, cat_j in enumerate(counts):
                value = counts[cat_j] / total  # n_k' / n
                if cat_i[1:] == cat_j[1:]:  # same sensitive attributes
                    value -= counts[cat_j] / s_tot[cat_i[1:]]  # n_k' / n_s
                if cat_i[0] != cat_j[0]:  # distinct label
                    value *= -1
                c_kk[idx, jdx] = value
        # Return the computed constants.
        return c_k0, c_kk


class EqualizedOddsFunction(FairnessFunction):
    """Equalized Odds group-fairness function.

    Definition
    ----------

    Equalized Odds is achieved when

    $$ \\forall l \\in Y, \\forall r \\in S,
       P(h(x) = y | y = l) == P(h(x) = y | y = l, s = r) $$

    where $Y$ denotes possible target labels, $S$ denotes possible values
    of a (set of intersected) sensitive attribute(s), and $h$ is the
    evaluated classifier.

    In other words, Equalized Odds is achieved when the probability that
    the model predicts the correct label is independent from the sensitive
    attribute(s).

    Formula
    -------

    For any sensitive group $k = (l, r)$ defined by the intersection of a
    given true label and a value of the sensitive attribute(s), Equalized
    Odds can be expressed in the canonical form from [1]:

    $$ F_k(h, T) = C_k^0 + \\sum_{k'} C_k^{k'} P(h(x) \\neq y | T_{k'}) $$

    using the following constants:

    - $ C_k^k = (n_k / n_l) - 1 $
    - $ C_k^{k'} = (n_{k'} / n_l) * 1{k_l = k'_l} $
    - $ C_k^0 = 0$

    where $n$ denotes a number of samples in the empirical dataset used,
    and its subscripted counterparts are number of samples that belong to
    a given sensitive group and/or have a given true label.

    References
    ----------
    [1] Maheshwari & Perrot (2023).
        FairGrad: Fairness Aware Gradient Descent.
        https://openreview.net/forum?id=0f8tU3QwWD
    """

    f_type = "equalized_odds"

    def compute_fairness_constants(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        counts = self._counts
        # Compute the label-wise number of samples.
        l_tot: Dict[Any, int] = {}  # label-wise number of samples
        for categ, count in counts.items():
            l_tot[categ[0]] = l_tot.get(categ[0], 0) + count
        # Compute fairness constraint constants C_k^k'.
        c_kk = np.zeros((len(counts), len(counts)))
        for idx, cat_i in enumerate(counts):
            for jdx, cat_j in enumerate(counts):
                if cat_i[0] != cat_j[0]:  # distinct target label
                    continue
                c_kk[idx, jdx] = counts[cat_j] / l_tot[cat_i[0]] - (idx == jdx)
        # Return the computed constants.
        c_k0 = np.array([0.0], dtype=c_kk.dtype)
        return c_k0, c_kk


class EqualityOfOpportunityFunction(EqualizedOddsFunction):
    """Equality of Opportunity group-fairness function.

    Definition
    ----------

    Equality of Opportunity is achieved when

    $$ \\forall l \\in Y' \\subset Y, \\forall r \\in S,
       P(h(x) = y | y = l) == P(h(x) = y | y = l, s = r) $$

    where $Y$ denotes possible target labels, $S$ denotes possible values
    of a (set of intersected) sensitive attribute(s), and $h$ is the
    evaluated classifier.

    In other words, Equality of Opportunity is equivalent to Equalized Odds
    but restricted to a subset of possible target labels. It is therefore
    achieved when the probability that the model predicts the correct label
    is independent from the sensitive attribute(s), for a subset of correct
    labels.

    Formula
    -------

    For any sensitive group $k = (l, r)$ defined by the intersection of a
    given true label and a value of the sensitive attribute(s), Equality
    of Opportunity can be expressed in the canonical form from [1]:

    $$ F_k(h, T) = C_k^0 + \\sum_{k'} C_k^{k'} P(h(x) \\neq y | T_{k'}) $$

    using the following constants:

    - For $k$ so that $k_l \\in Y'$:
        - $ C_k^k = (n_k / n_l) - 1 $
        - $ C_k^{k'} = (n_{k'} / n_l) * 1{k_l = k'_l} $
    - All other constants are null.

    where $n$ denotes a number of samples in the empirical dataset used,
    and its subscripted counterparts are number of samples that belong to
    a given sensitive group and/or have a given true label.

    References
    ----------
    [1] Maheshwari & Perrot (2023).
        FairGrad: Fairness Aware Gradient Descent.
        https://openreview.net/forum?id=0f8tU3QwWD
    """

    f_type = "equality_of_opportunity"

    def __init__(
        self,
        counts: Dict[Tuple[Any, ...], int],
        target: Union[int, List[int]] = 1,
    ) -> None:
        """Instantiate an Equality of Opportunity group-fairness function.

        Parameters
        ----------
        counts:
            Group-wise counts for all target label and sensitive attribute(s)
            intersected values, with format `{(label, *attrs): count}`.
        target:
            Label(s) that fairness constraints are to be restricted to.
        """
        # Parse 'target' inputs.
        if isinstance(target, int):
            self._target = {target}
        elif isinstance(target, (list, tuple, set)):
            self._target = set(target)
        else:
            raise TypeError("'target' should be an int or list of ints.")
        # Verify that 'target' is a subset of target labels.
        targets = set(int(group[0]) for group in counts)
        if not self._target.issubset(targets):
            raise ValueError(
                "'target' should be a subset of target label values present "
                "in sensitive groups' definitions."
            )
        # Delegate remainder of instantiation to the parent class.
        super().__init__(counts=counts)

    def compute_fairness_constants(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Compute equalized odds constants (as if all targets were desirable).
        c_k0, c_kk = super().compute_fairness_constants()
        # Zero-out constants associated with undesired targets.
        for idx, cat_i in enumerate(self._counts):
            if int(cat_i[0]) not in self._target:
                c_kk[idx] = 0.0
        # Return the computed constants.
        return c_k0, c_kk

    def get_specs(
        self,
    ) -> Dict[str, Any]:
        specs = super().get_specs()
        specs["target"] = list(self._target)
        return specs
