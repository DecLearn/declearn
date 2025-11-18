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

"""ABC and generic constructor for group-fairness functions."""

import abc
import functools
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np

from declearn.utils import (
    access_registered,
    create_types_registry,
    register_type,
)

__all__ = [
    "FairnessFunction",
    "instantiate_fairness_function",
]


@create_types_registry(name="FairnessFunction")
class FairnessFunction(metaclass=abc.ABCMeta):
    """Abstract base class for group-fairness functions.

    Group Fairness
    --------------

    This abstract base class defines a common API and shared backend code
    to compute group-wise fairness metrics for group-fairness definitions
    that can be written using the following canonical form, introduced by
    Maheshwari & Perrot (2023) [1]:

    $$ F_k(h, T) = C_k^0 + \\sum_{k'} C_k^{k'} P(h(x) \\neq y | T_{k'}) $$

    Where $F_k$ is the fairness metric associated with sensitive group $k$,
    $h$ is the evaluated classifier, $T$ denotes a data distribution (that
    is approximated based on an empirical dataset) and $T_k$ denotes the
    distribution of samples belonging to group $k$.

    Scope
    -----

    This class implements the formula above based on empirical sample counts
    and group-wise accuracy estimates. It also implements its adaptation to
    the federated learning setting, where $P(h(x) \\neq y | T_{k'})$ can be
    computed from client-wise accuracy values for local samples belonging to
    group $k'$, as their weighted average based on client-wise group-wise
    sample counts. Here, clients merely need to send values scaled by their
    local counts, while the server only needs to access the total group-wise
    counts, which makes these computations compatible with SecAgg.

    This class does not implement the computation of group-wise counts,
    nor the evaluation of the group-wise accuracy of a given model, but
    merely instruments these quantities together with the definition of
    a group-fairness notion in order to compute the latter's values.

    Inheritance
    -----------

    Subclasses of `FairnessFunction` are required to:

    - specify a `f_type` string class attribute, which is meant to be
      unique across subclasses;
    - implement the `compute_fairness_constants` abstract method, which
      is called upon instantiation to define the $C_k^{k'}$ constants
      from the canonical form of the group-fairness function.

    By default, subclasses are type-registered under the `f_type` name,
    enabling instantiation with the `instantiate_fairness_function` generic
    constructor. This can be disabled by passing the `register=False` kwarg
    at inheritance (e.g. `class MyFunc(FairnessFunction, register=False):`).

    References
    ----------
    [1] Maheshwari & Perrot (2023).
        FairGrad: Fairness Aware Gradient Descent.
        https://openreview.net/forum?id=0f8tU3QwWD
    """

    f_type: ClassVar[str]

    def __init__(
        self,
        counts: Dict[Tuple[Any, ...], int],
    ) -> None:
        """Instantiate a group-fairness function from its defining parameters.

        Parameters
        ----------
        counts:
            Group-wise counts for all target label and sensitive attribute(s)
            intersected values, with format `{(label, *attrs): count}`.
        """
        self._counts = counts.copy()
        c_k0, c_kk = self.compute_fairness_constants()
        self._ck0 = c_k0
        self._ckk = c_kk

    def __init_subclass__(
        cls,
        register: bool = True,
    ) -> None:
        """Automatically type-register subclasses."""
        if register:
            register_type(cls, name=cls.f_type, group="FairnessFunction")

    @functools.cached_property
    def groups(self) -> List[Tuple[Any, ...]]:
        """Sorted list of defined sensitive groups for this function."""
        return sorted(self._counts)

    @functools.cached_property
    def constants(self) -> Tuple[np.ndarray, np.ndarray]:
        """Constants defining the fairness function."""
        return self._ck0.copy(), self._ckk.copy()

    @abc.abstractmethod
    def compute_fairness_constants(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute fairness constants associated with this function.

        This method computes and returns fairness constants that are based
        on the specific group-fairness definition, on the sensitive groups'
        sample counts and on any other class-specific hyper-parameter set
        at instantiation.

        This method is notably called upon instantiation, to produce values
        that are instrumental in all subsequent fairness computations using
        the defined setting.

        Cached values may be accessed using the `constants` property getter.

        Returns
        -------
        c_k0:
            1-d array containing $C_k^0$ constants for each and every group.
            May be a single-value array, notably when $C_k^0 = 0 \\forall k$.
        c_kk:
            2-d array containing $C_k^{k'}$ constants for each and every pair
            of sensitive groups.

        Raises
        ------
        ValueError
            If the fairness constants' computation fails.
        """

    def compute_from_group_accuracy(
        self,
        accuracy: Dict[Tuple[Any, ...], float],
    ) -> Dict[Tuple[Any, ...], float]:
        """Compute the fairness function from group-wise accuracy metrics.

        Parameters
        ----------
        accuracy:
            Group-wise accuracy values of the model being evaluated on a
            dataset. I.e. `{group_k: P(y_pred == y_true | group_k)}`.

        Returns
        -------
        fairness:
            Group-wise fairness metrics,  as a `{group_k: score_k}` dict.
            Values' interpretation depend on the implemented group-fairness
            definition, but overall the fairer the accuracy towards a group,
            the closer the metric is to zero.

        Raises
        ------
        KeyError
            If any defined sensitive group does not have an accuracy metric.
        """
        if diff := set(self.groups).difference(accuracy):
            raise KeyError(
                f"Group accuracies from groups {diff} are missing from inputs"
                f" to '{self.__class__.__name__}.compute_from_group_accuracy'."
            )
        # Compute F_k = C_k^0 + Sum_{k'}(C_k^k' * (1 - acc_k'))
        cerr = 1 - np.array([accuracy[group] for group in self.groups])
        c_k0, c_kk = self.constants
        f_k = c_k0 + np.dot(c_kk, cerr)
        # Wrap up results as a {group: score} dict, for readability purposes.
        return dict(zip(self.groups, f_k.tolist(), strict=False))

    def compute_from_federated_group_accuracy(
        self,
        accuracy: Dict[Tuple[Any, ...], float],
    ) -> Dict[Tuple[Any, ...], float]:
        """Compute the fairness function from federated group-wise accuracy.

        Parameters
        ----------
        accuracy:
            Group-wise sum-aggregated local-group-count-weighted accuracies
            of a given model over an ensemble of local datasets.
            I.e. `{group_k: sum_i(n_ik * accuracy_ik)}`.

        Returns
        -------
        fairness:
            Group-wise fairness metrics,  as a `{group_k: score_k}` dict.
            Values' interpretation depend on the implemented group-fairness
            definition, but overall the fairer the accuracy towards a group,
            the closer the metric is to zero.

        Raises
        ------
        KeyError
            If any defined sensitive group does not have an accuracy metric.
        """
        accuracy = {
            key: val / cnt
            for key, val in accuracy.items()
            if (cnt := self._counts.get(key)) is not None
        }
        return self.compute_from_group_accuracy(accuracy)

    def get_specs(
        self,
    ) -> Dict[str, Any]:
        """Return specifications of this fairness function.

        Returns
        -------
        specs:
            Dict of keyword arguments that may be passed to the
            `declearn.fairness.core.instantiate_fairness_function`
            generic constructor to recover a copy of this instance.
        """
        return {"f_type": self.f_type, "counts": self._counts.copy()}


def instantiate_fairness_function(
    f_type: str,
    counts: Dict[Tuple[Any, ...], int],
    **kwargs: Any,
) -> FairnessFunction:
    """Instantiate a FairnessFunction from its specifications.

    Parameters
    ----------
    f_type:
        Name of the type of group-fairness function to instantiate.
    counts:
        Group-wise counts for all target label and sensitive attribute(s)
        intersected values, with format `{(label, *attrs): count}`.
    **kwargs:
        Any keyword argument for the instantiation of the target function
        may be passed.
    """
    cls = access_registered(name=f_type, group="FairnessFunction")
    assert issubclass(cls, FairnessFunction)
    return cls(counts=counts, **kwargs)
