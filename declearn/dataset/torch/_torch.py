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

"""Dataset implementation to serve torch datasets."""

import dataclasses
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch

from declearn.dataset._base import Dataset, DataSpecs
from declearn.dataset.torch._utils import PoissonSampler
from declearn.typing import Batch
from declearn.utils import register_type

__all__ = [
    "TorchDataset",
]


@register_type(group="Dataset")
class TorchDataset(Dataset):
    """Dataset subclass serving torch Datasets.

    This subclass implements:

    * yielding (X, [y], [w]) batches matching the expected batch
      format, with each elements being either a torch.tensor,
      an iterable of torch.tensors, or None
    * loading the source data from which batches are derived
      using the provided torch.dataset
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        collate_fn: Optional[
            Callable[
                [List[Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]]],
                Tuple[Union[List[torch.Tensor], torch.Tensor], ...],
            ]
        ] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Instantiate a declearn Dataset wrapping a torch.utils.data.Dataset.

        Instantiate the declearn dataset interface from an existing
        torch.utils.data.Dataset object. Minimal checks run on the user
        provided torch.utils.data.Dataset, so in case of errors,
        the user is expected to refer to the documention for guidance.

        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            An torch Dataset instance built by the user, to be wrapped in
            declearn. The dataset's `__getitem__` method is expected to
            return either a single torch.Tensor (the model inputs) or a
            tuple of (model inputs, optional label, optional sample weights)
            as torch.Tensors or list of torch.Tensors.
        collate_fn: callable or None, default=None
            Optional collate function to merge a list of samples (formatted
            as tuples of tensors and/or lists of tensors) into a mini-batch.
            If None, use `torch.utils.data.default_collate`.
        seed: int or None, default=None
            Optional seed for the random number generator based on which
            the dataset is (optionally) shuffled when generating batches.

        Notes
        -----
        The wrapped `torch.utils.data.Dataset`:

        - *must* implement the `__len__` method, defining its size.
        - *may* implement a `get_data_specs` method, returning metadata
          that are to be shared with the FL server, as a dict with keys
          and types that match the `declearn.dataset.DataSpecs` fields.
        - should return sample-level (unbatched) elements, as either:
            - inputs
            - (inputs,)
            - (inputs, labels)
            - (inputs, labels, weights)
          where:
            - inputs may be a single tensor or list of tensors
            - labels may be a single tensor or None
            - weights may be a single tensor or None

        When dealing with data that requires specific processing to be
        batched (e.g. some sort of padding), please use a `collate_fn`
        to define that processing. For samples that all share the same
        shape, the default collate function should suffice.
        """
        super().__init__()
        self.dataset = dataset
        if collate_fn is None:
            collate_fn = torch.utils.data.default_collate
        self.collate_fn = collate_fn
        # Assign a random number generator.
        self.seed = seed
        self.gen: Optional[torch.Generator] = None
        if self.seed is not None:
            # pylint: disable=no-member
            self.gen = torch.Generator().manual_seed(self.seed)

    def _get_length(self) -> int:
        """Access the wrapped torch Dataset's length, raising if undefined."""
        try:
            return len(self.dataset)  # type: ignore
        except TypeError as exc:
            raise TypeError(
                "'TorchDataset' requires the input dataset to implement the "
                "'__len__' method to expose its size."
            ) from exc

    def get_data_specs(
        self,
    ) -> DataSpecs:
        """Return a DataSpecs object describing this dataset."""
        specs: Dict[str, Any] = {"n_samples": self._get_length()}
        if hasattr(self.dataset, "get_data_specs"):
            user_specs = self.dataset.get_data_specs()
            if isinstance(user_specs, dict):
                self.check_dataset_specs(user_specs)
                specs.update(user_specs)
        return DataSpecs(**specs)

    # pylint: disable-next=too-many-positional-arguments
    def generate_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_remainder: bool = True,
        replacement: bool = False,
        poisson: bool = False,
    ) -> Iterator[Batch]:
        """Yield batches of data samples.

        Parameters
        ----------
        batch_size: int
            Number of samples per batch.
            If `poisson=True`, this is the average batch size.
        shuffle: bool, default=False
            Whether to shuffle data samples prior to batching.
            Note that the shuffling will differ on each call
            to this method.
        drop_remainder: bool, default=True
            Whether to drop the last batch if it contains less
            samples than `batch_size`, or yield it anyway.
            If `poisson=True`, this is used to determine the number
            of returned batches (notwithstanding their actual size).
        replacement: bool, default=False
            Whether to do random sampling with or without replacement.
            Ignored if `shuffle=False` or `poisson=True`.
        poisson: bool, default=False
            Whether to use Poisson sampling, i.e. make up batches by
            drawing samples with replacement, resulting in variable-
            size batches and samples possibly appearing in zero or in
            multiple emitted batches (but at most once per batch).
            Useful to maintain tight Differential Privacy guarantees.

        Yields
        ------
        inputs: torch.Tensor or list(torch.Tensor)
            Input features.
        targets: torch.Tensor or list(torch.Tensor) or None
            Optional target labels or values.
        weights: torch.Tensor or None
            Optional sample weights.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        if poisson:
            n_samples = self._get_length()
            batch_sampler: torch.utils.data.Sampler = PoissonSampler(
                num_samples=n_samples,
                sample_rate=batch_size / n_samples,
                generator=self.gen,
            )
        else:
            if shuffle:
                sampler: torch.utils.data.Sampler = (
                    torch.utils.data.RandomSampler(
                        data_source=self.dataset,  # type: ignore
                        # sized Dataset
                        replacement=replacement,
                        generator=self.gen,
                    )
                )
            else:
                sampler = torch.utils.data.SequentialSampler(
                    data_source=self.dataset  # type: ignore  # sized Dataset
                )
            batch_sampler = torch.utils.data.BatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                drop_last=drop_remainder,
            )
        yield from torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_to_batch,
        )

    @staticmethod
    def check_dataset_specs(specs) -> None:
        """Utility function checking that user-defined `get_specs()`
        method returns valid [DataSpecs][declearn.dataset.DataSpecs]
        fields."""
        acceptable = {f.name for f in dataclasses.fields(DataSpecs)}
        for key in specs.keys():
            if key not in acceptable:
                raise ValueError(
                    "All keys of the dictionnary returned by your original"
                    " Torch Dataset method `get_specs()` must conform to one"
                    "of the fields found in `declearn.dataset.DataSpecs`."
                    f"'{key}' did not. "
                )

    def collate_to_batch(
        self,
        samples: Union[
            List[Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]],
            List[Union[torch.Tensor, List[torch.Tensor]]],
        ],
    ) -> Tuple[
        Union[List[torch.Tensor], torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Custom collate method to structure samples into a batch.

        This method wraps up the `collate_fn` attribute of this instance
        (which, by default, is `torch.utils.data.default_collate`) so as
        to take into account the declearn specs about the input data and
        output batches' formatting.

        Parameters
        ----------
        samples:
            List of sample elements that are to be collated into a batch.
            Each sample may either be:

            - a single Tensor or list of Tensors, denoting input data
            - a tuple containing 1 to 3 (lists of) Tensors, denoting,
              in that order: input data, target labels and sample
              weights.

        Returns
        -------
        batch:
            Batch of (x, y, w) stacked samples, where x may be a list,
            and y and w may be None.
        """
        if not isinstance(samples[0], tuple):
            samples = [(sample,) for sample in samples]  # type: ignore
        batch = self.collate_fn(samples)  # type: ignore
        if not 1 <= len(batch) <= 3:
            raise TypeError(
                "Raw batches should contain 1 to 3 elements, denoting (in "
                "that order) model inputs, true labels and sample weights."
            )
        padding = [None] * (3 - len(batch))
        return (*batch, *padding)  # type: ignore
