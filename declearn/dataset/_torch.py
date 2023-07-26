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

"""Dataset implementation to serve torch datasets."""

import dataclasses
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch

from declearn.dataset._base import Dataset, DataSpecs
from declearn.typing import Batch
from declearn.utils import register_type

TorchBatch = Tuple[
    Union[List[torch.Tensor], torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
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
        target: data array or str or None, default=None
            Optional data array containing target labels (for supervised
            learning)
        s_wght: int or str or function or None, default=None
            Optional data array containing sample weights
        seed: int or None, default=None
            Optional seed for the random number generator based on which
            the dataset is (optionally) shuffled when generating batches.
        """
        super().__init__()
        self.dataset = dataset
        self.data_item_info = get_data_item_info(self.dataset[0])
        # Assign a random number generator.
        self.seed = seed
        self.gen = None
        self.my_collate = self.get_custom_collate(self.data_item_info)
        if self.seed is not None:
            torch.manual_seed(self.seed)
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
        specs = {"n_samples": self._get_length()}  # type: Dict[str, Any]
        if hasattr(self.dataset, "get_data_specs"):
            user_specs = self.dataset.get_data_specs()
            if isinstance(user_specs, dict):
                self.check_dataset_specs(user_specs)
                specs.update(user_specs)
        return DataSpecs(**specs)

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
            try:
                # conditional import; pylint: disable=import-outside-toplevel
                from opacus.utils.uniform_sampler import (  # type: ignore
                    UniformWithReplacementSampler,
                )
            except ModuleNotFoundError as exc:
                # pragma: no cover
                raise ImportError(
                    "Cannot use Poisson sampling on 'TorchDataset': "
                    "missing optional dependency 'opacus', required "
                    "for this feature."
                ) from exc
            n_samples = self._get_length()
            batch_sampler = UniformWithReplacementSampler(
                num_samples=n_samples,
                sample_rate=batch_size / n_samples,
                generator=self.gen,
            )
        else:
            if shuffle:
                sampler = torch.utils.data.RandomSampler(
                    data_source=self.dataset,  # type: ignore  # sized Dataset
                    replacement=replacement,
                    generator=self.gen,
                )  # type: torch.utils.data.Sampler
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
            collate_fn=self.my_collate,
        )

    @staticmethod
    def check_dataset_specs(specs) -> None:
        """Utility function checking that user-defined `get_specs()`
        method returns valid [DataSpecs][declearn.dataset.Dataspecs]
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

    @staticmethod
    def get_custom_collate(data_item_info: Dict[str, Any]) -> Callable:
        """Given the type and lenght of the items returned by
        `self.dataset.__getitem__()`, returns the appropriate utility function
        to cast the data items to the expected output format. This function
        is meant to be used  as an argument to the torch Dataloader.

        """
        return partial(transform_batch, data_item_info=data_item_info)


# Utility functions


def get_data_item_info(data_item) -> Dict[str, Any]:
    """Check that the user-defined dataset `__getitem__` method returns a
    data_item that is easily castable to the format expected for
    declearn-based optimization. If not, raises an error.

    Note : we assume that if the dataset returns a tuple of tensors of
    the form (input,label,weights). The edge case with no labels provided
    but the sample weights is not currently covered."""

    if isinstance(data_item, torch.Tensor):
        return {"type": torch.Tensor, "len": None}
    if isinstance(data_item, tuple) and 0 < len(data_item) < 4:
        return {"type": tuple, "len": len(data_item)}
    raise ValueError(
        "Your orignal Torch Dataset should return either a single"
        "torch.Tensor (the model inputs) or a tuple of (model"
        "inputs, optional label, optional sample weights) as torch"
        ".Tensors"
    )


def transform_batch(batch, data_item_info) -> TorchBatch:
    """
    Based on the type and shape of data items, returns a tuple of lenght
    three containing the batched data and patched with Nones.
    """
    original_batch = torch.utils.data.default_collate(batch)
    out: TorchBatch
    if data_item_info["type"] == torch.Tensor:
        out = (original_batch, None, None)
    else:
        patch = [None for _ in range(3 - data_item_info["len"])]
        out = (*original_batch, *patch)  # type: ignore
    return out
