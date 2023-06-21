import dataclasses
import functools
import importlib
import os
import warnings
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data import Dataset as OriginalTorchDataset
from torch.utils.data import RandomSampler, Sampler, SequentialSampler

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
    """wrapping torch dataset"""

    _type_key: ClassVar[str] = "TorchDataset"

    def __init__(
        self,
        dataset: OriginalTorchDataset,
        seed: Optional[int] = None,
    ) -> None:
        """

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
            tuple of (model input, optional label, optional sample weights)
            as torch.Tensors or list of torch.Tensors".
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
        self.dataset = self.validate_dataset(dataset)
        # Assign a random number generator.
        self.seed = seed
        self.gen = None
        if self.seed is not None:
            torch.manual_seed(self.seed)
            self.gen = torch.Generator().manual_seed(seed)

    def get_data_specs(
        self,
    ) -> DataSpecs:
        """Return a DataSpecs object describing this dataset."""
        specs = {"n_samples": len(self.dataset)}
        run_cond = hasattr(self.dataset, "get_specs")
        if run_cond and isinstance(self.dataset.get_specs(), dict):
            self.check_dataset_specs(specs)
            specs.update(self.dataset.specs)
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
            Whether to do randopm sampling with or without replacement.
            Ignored if shuffle = False or poisson = True.
        poisson: bool, default=False
            Whether to use Poisson sampling, i.e. make up batches by
            drawing samples with replacement, resulting in variable-
            size batches and samples possibly appearing in zero or in
            multiple emitted batches (but at most once per batch).
            Useful to maintain tight Differential Privacy guarantees.

        Yields
        ------
        inputs: torch.Tensor or list(torch.Tensor)
            Input features
        targets: torch.Tensor or None
            Optional target labels or values
        weights: torch.Tensor or None
            Optional sample weights

        """
        sampler: Sampler
        if poisson:
            module = "opacus.utils.UniformWithReplacementSampler"
            sampler_class = importlib.import_module(module)
            n_samples = len(self.dataset)
            rate = batch_size / n_samples
            batch_sampler = sampler_class(
                num_samples=n_samples,
                sample_rate=rate,
                generator=self.gen,
            )
        else:
            if shuffle:
                sampler = RandomSampler(
                    data_source=self.dataset,
                    replacement=replacement,
                    generator=self.gen,
                )
            else:
                sampler = SequentialSampler(data_source=self.dataset)
            batch_sampler = BatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                drop_last=drop_remainder,
            )
        yield from DataLoader(
            dataset=self.dataset, batch_sampler=batch_sampler
        )

    @staticmethod
    def check_dataset_specs(specs) -> None:
        """Utility function checking that user-defined `get_specs()`
        method returns valid [DataSpecs][declearn.dataset.Dataspecs]
        fields."""
        acceptable = {f.name for f in dataclasses.fields(DataSpecs)}
        for key, _ in specs:
            if key not in acceptable:
                raise ValueError(
                    "All keys of the dictionnary returned by your original Torch"
                    "Dataset method `get_specs()` must conform to one of the fields"
                    f"found in `declearn.dataset.DataSpecs`. '{key}' did not. "
                )

    @staticmethod
    def validate_dataset(
        dataset: OriginalTorchDataset,
    ) -> OriginalTorchDataset:
        """Check that the user-defined dataset `__getitem__` method returns a
        valid input for declearn-based optimization, and if not overides that
        method. Some corner cases not covered.

        Note : the dataset needs to return a something easily castable to a
        [Batch][declearn.typing.Batch].
        """

        def transform(data_item: Any) -> TorchBatch:
            if isinstance(data_item, torch.Tensor):
                out = (data_item, None, None)
            elif isinstance(data_item, tuple):
                if len(data_item) == 1:
                    out = (data_item, None, None)
                elif len(data_item) == 2:
                    out = (data_item[0], data_item[1], None)
                elif len(data_item) == 3:
                    out = data_item
            else:
                raise ValueError(
                    "Your orignal Torch Dataset should return either a single"
                    "torch.Tensor (the model inputs) or a tuple of (model"
                    "input, optional label, optional sample weights) as torch"
                    ".Tensors"
                )
            return out

        def new_getitem(*args):
            out = getitem(*args)
            return transform(out)

        getitem = getattr(dataset, "__getitem__")
        setattr(dataset, "__getitem__", new_getitem)
        return dataset
