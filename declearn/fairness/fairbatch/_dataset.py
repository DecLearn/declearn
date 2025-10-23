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

"""FairBatch-specific Dataset wrapper and subclass."""

from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np

from declearn.dataset import Dataset, DataSpecs
from declearn.fairness.api import FairnessDataset
from declearn.typing import Batch

__all__ = [
    "FairbatchDataset",
]


class FairbatchDataset(FairnessDataset):
    """FairBatch-specific FairnessDataset subclass and wrapper.

    FairBatch is an algorithm to enforce group fairness when learning
    a classifier, initially designed for the centralized setting, and
    extendable to the federated one. It mostly relies on changing the
    way training data batches are drawn: instead of drawing uniformly
    from the full dataset, FairBatch introduces sampling probabilities
    attached to the sensitive groups, that are updated throughout time
    to reflect the model's current fairness levels.

    This class is both a subclass to `FairnessDataset` and a wrapper
    that is designed to hold such a dataset. It implements a couple
    of algorithm-specific methods to set or get group-wise sampling
    probabilities, and transparently introduces the FairBatch logic
    into the API-defined `generate_batches` method.

    This implementation is based both on the original FairBatch paper
    and on the reference implementation by the paper's authors. Hence,
    instead of effectively assigning drawing probabilities to samples
    based on their sensitive group, batches are in fact drawn as the
    concatenation of fixed-size sub-batches, drawn from data subsets
    defined by samples' sensitive group.

    As in the reference implementation:

    - ouptut batches always have the same number of samples - this is
      true even when using `drop_remainder=False`, that merely adds a
      batch to the sequence of generated batches;
    - the number of batches to yield is computed based on the total
      number of samples and full abtch size;
    - when a subset is exhausted, it is drawn from anew; hence, samples
      may be seen multiple time in a single "epoch" depending on the
      groups' number of samples and sampling probabilities;
    - in the extreme case when a subset is smaller than the number of
      samples that should be drawn from it for any batch, samples may
      even be included multiple times in the same batch.

    In the federated setting, clients may not hold samples to each and
    every sensitive group. In this implementation, when a client has no
    samples for a given group, it adjusts the sampling probabilities of
    all groups for which they have samples. In other words, sampling
    probabilities are adjusted so that the total batch size is the same
    across clients, in spite of some clients possibly not having samples
    for some groups.
    """

    def __init__(
        self,
        base: FairnessDataset,
    ) -> None:
        """Instantiate a FairbatchDataset wrapping a FairnessDataset.

        Parameters
        ----------
        base:
            Base `FairnessDataset` instance to wrap so as to apply
            group-wise subsampling as per the FairBatch algorithm.
        """
        self.base = base
        # Assign a dictionary with sampling probability for each group.
        self.groups = self.base.get_sensitive_group_definitions()
        self._counts = self.base.get_sensitive_group_counts()
        self._sampling_probas = {
            group: 1.0 / len(self.groups) for group in self.groups
        }

    # Methods provided by the wrapped dataset (merely interfaced).

    def get_data_specs(
        self,
    ) -> DataSpecs:
        return self.base.get_data_specs()

    def get_sensitive_group_definitions(
        self,
    ) -> List[Tuple[Any, ...]]:
        return self.groups

    def get_sensitive_group_counts(
        self,
    ) -> Dict[Tuple[Any, ...], int]:
        return self._counts.copy()

    def get_sensitive_group_subset(
        self,
        group: Tuple[Any, ...],
    ) -> Dataset:
        return self.base.get_sensitive_group_subset(group)

    def set_sensitive_group_weights(
        self,
        weights: Dict[Tuple[Any, ...], float],
        adjust_by_counts: bool = False,
    ) -> None:
        self.base.set_sensitive_group_weights(weights, adjust_by_counts)

    # FairBatch-specific methods.

    def get_sampling_probabilities(
        self,
    ) -> Dict[Tuple[Any, ...], float]:
        """Access current group-wise sampling probabilities."""
        return self._sampling_probas.copy()

    def set_sampling_probabilities(
        self,
        group_probas: Dict[Tuple[Any, ...], float],
    ) -> None:
        """Assign new group-wise sampling probabilities.

        If some groups are not present in the wrapped dataset,
        scale the probabilities associated with all represented
        groups so that they sum to 1.

        Parameters
        ----------
        group_probas:
            Dict of group-wise sampling probabilities, with
            `{(s_attr_1, ..., s_attr_k): sampling_proba}` format.

        Raises
        ------
        ValueError
            If the input probabilities are not positive values
            or if they do not cover (a superset of) all sensitive
            groups present in the wrapped dataset.
        """
        # Verify that input match expectations.
        if not all(x >= 0 for x in group_probas.values()):
            raise ValueError(
                f"'{self.__class__.__name__}.update_sampling_probabilities' "
                "cannot have a negative probability value as parameter."
            )
        if not set(self.groups).issubset(group_probas):
            raise ValueError(
                "'FairbatchDataset.update_sampling_probabilities' requires "
                "input values to cover (a superset of) local sensitive groups."
            )
        # Restrict and adjust probabilities to groups with samples.
        probas = {group: group_probas[group] for group in self.groups}
        total = sum(probas.values())
        self._sampling_probas = {
            key: val / total for key, val in probas.items()
        }

    # pylint: disable-next=too-many-positional-arguments
    def generate_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_remainder: bool = True,
        replacement: bool = False,
        poisson: bool = False,
    ) -> Iterator[Batch]:
        # inherited signature; pylint: disable=too-many-arguments
        # Compute the number of batches to yield.
        nb_samples = sum(self._counts.values())
        nb_batches = nb_samples // batch_size
        if (not drop_remainder) and (nb_samples % batch_size):
            nb_batches += 1
        # Compute the group-wise number of samples per batch.
        # NOTE: this number may be reduced if there are too few samples.
        group_batch_size = {
            group: round(proba * batch_size)
            for group, proba in self._sampling_probas.items()
        }
        # Yield batches made of a fixed number of samples from each group.
        generators = [
            self._generate_sensitive_group_batches(
                group, nb_batches, g_batch_size, shuffle, replacement, poisson
            )
            for group, g_batch_size in group_batch_size.items()
            if g_batch_size > 0
        ]
        for batches in zip(*generators, strict=False):
            yield self._concatenate_batches(batches)

    @staticmethod
    def _concatenate_batches(
        batches: Sequence[Batch],
    ) -> Batch:
        """Concatenate batches of numpy array data."""
        x_dat = np.concatenate([batch[0] for batch in batches], axis=0)
        y_dat = (
            None
            if batches[0][1] is None
            else np.concatenate([batch[1] for batch in batches], axis=0)
        )
        w_dat = (
            None
            if batches[0][2] is None
            else np.concatenate([batch[2] for batch in batches], axis=0)
        )
        return x_dat, y_dat, w_dat

    # pylint: disable-next=too-many-positional-arguments
    def _generate_sensitive_group_batches(  # noqa: PLR0913
        self,
        group: Tuple[Any, ...],
        nb_batches: int,
        batch_size: int,
        shuffle: bool,
        replacement: bool,
        poisson: bool,
    ) -> Iterator[Batch]:
        """Generate a fixed number of batches for a given sensitive group.

        Parameters
        ----------
        group:
            Sensitive group, the dataset from which to draw from.
        nb_batches:
            Number of batches to yield. The dataset will be iterated
            over if needed to achieve this number.
        batch_size:
            Number of samples per batch (will be exact).
        shuffle:
            Whether to shuffle the dataset prior to drawing batches.
        replacement:
            Whether to draw with replacement between batches.
        poisson:
            Whether to use poisson sampling rather than batching.
        """
        # backend method; pylint: disable=too-many-arguments
        args = (shuffle, replacement, poisson)
        # Fetch the target sub-dataset and its samples count.
        dataset = self.get_sensitive_group_subset(group)
        n_samples = self._counts[group]
        # When the dataset is large enough, merely yield batches.
        if batch_size <= n_samples:
            yield from self._generate_batches(
                dataset, group, nb_batches, batch_size, *args
            )
        # When the batch size is larger than the number of data points,
        # make up a base batch will all points (duplicated if needed),
        # that will be combined with further batches of data.
        else:
            n_repeats, batch_size = divmod(batch_size, n_samples)
            # Gather the full subset, optionally duplicated.
            full = self._get_full_dataset(dataset, n_samples, group)
            if n_repeats > 1:
                full = self._concatenate_batches([full] * n_repeats)
            # Add up further (batch-varying) samples (when needed).
            if batch_size:
                for batch in self._generate_batches(
                    dataset, group, nb_batches, batch_size, *args
                ):
                    yield self._concatenate_batches([full, batch])
            else:  # edge case: require exactly N times the full dataset
                for _ in range(nb_batches):
                    yield full

    # pylint: disable-next=too-many-positional-arguments
    def _generate_batches(  # noqa: PLR0913
        self,
        dataset: Dataset,
        group: Tuple[Any, ...],
        nb_batches: int,
        batch_size: int,
        shuffle: bool,
        replacement: bool,
        poisson: bool,
    ) -> Iterator[Batch]:
        """Backend to yield a fixed number of batches from a dataset."""
        # backend method; pylint: disable=too-many-arguments
        # Iterate multiple times over the sub-dataset if needed.
        counter = 0
        while counter < nb_batches:
            # Yield batches from the sub-dataset.
            generator = dataset.generate_batches(
                batch_size=batch_size,
                shuffle=shuffle,
                drop_remainder=True,
                replacement=replacement,
                poisson=poisson,
            )
            for batch in generator:
                yield batch
                counter += 1
                if counter == nb_batches:
                    break
            # Prevent infinite loops and raise an informative error.
            if not counter:  # pragma: no cover
                raise RuntimeError(
                    f"'{self.__class__.__name__}.generate_batches' triggered "
                    "an infinite loop; this happened when trying to extract "
                    f"{batch_size}-samples batches for group {group}."
                )

    @staticmethod
    def _get_full_dataset(
        dataset: Dataset,
        n_samples: int,
        group: Tuple[Any, ...],
    ) -> Batch:
        """Return a batch containing an entire dataset's samples."""
        try:
            generator = dataset.generate_batches(
                batch_size=n_samples,
                shuffle=False,
                drop_remainder=False,
                replacement=False,
                poisson=False,
            )
            return next(generator)
        except StopIteration as exc:  # pragma: no cover
            raise RuntimeError(
                f"Failed to fetch the full subdataset for group '{group}'."
            ) from exc
