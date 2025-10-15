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

"""Fairness-aware Dataset abstract base subclass."""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

from declearn.dataset import Dataset

__all__ = [
    "FairnessDataset",
]


class FairnessDataset(Dataset, metaclass=ABCMeta):
    """Abstract base class for Fairness-aware Dataset interfaces.

    This `declearn.dataset.Dataset` abstract subclass adds API methods
    related to group fairness to the base dataset API. These revolve
    around accessing sensitive group definitions, sample counts and
    dataset subset. They further add the possibility to modify samples'
    weights based on the sensitive group to which they belong.
    """

    @abstractmethod
    def get_sensitive_group_definitions(
        self,
    ) -> List[Tuple[Any, ...]]:
        """Return a list of exhaustive sensitive groups for this dataset.

        Returns
        -------
        groups:
            List of tuples of values that define a sensitive group, as
            the intersection of one or more sensitive attributes, and
            the model's target when defined.
        """

    @abstractmethod
    def get_sensitive_group_counts(
        self,
    ) -> Dict[Tuple[Any, ...], int]:
        """Return sensitive attributes' combinations and counts.

        Returns
        -------
        values:
            Dict holding the number of samples for each and every sensitive
            group. Its keys are tuples holding the values of the attributes
            that define the sensitive groups (based on their intersection).
        """

    @abstractmethod
    def get_sensitive_group_subset(
        self,
        group: Tuple[Any, ...],
    ) -> Dataset:
        """Return samples that belong to a given sensitive group.

        Parameters
        ----------
        group:
            Tuple of values that define a sensitive group (as samples that
            have these values as sensitive attributes and/or target label).

        Returns
        -------
        dataset:
            `Dataset` instance, holding the subset of samples that belong
            to the specified sensitive group.

        Raises
        ------
        KeyError
            If the specified group does not match any sensitive group
            defined for this instance.
        """

    @abstractmethod
    def set_sensitive_group_weights(
        self,
        weights: Dict[Tuple[Any, ...], float],
        adjust_by_counts: bool = False,
    ) -> None:
        """Assign weights associated with samples' sensitive group membership.

        This method updates the sample weights yielded by this dataset's
        `generate_batches` method, to become the product of the raw sample
        weights and the values associated with the sensitive attributes.

        Parameters
        ----------
        weights:
            Dict associating weights with tuples of values caracterizing the
            sensitive groups they are to apply to.
        adjust_by_counts:
            Whether to multiply input group weights `w_k` by the number of
            samples for their group. This is notably useful in federated
            contexts, where `weights` may in fact be input as `w_k^t / n_k`
            and thereof adjusted to `w_k^t * n_{i,k} / n_k`.

        Raises
        ------
        KeyError
            If not all local sensitive groups have weights defined as part
            of the inputs.
        """
