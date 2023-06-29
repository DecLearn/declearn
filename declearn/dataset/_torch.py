import dataclasses
import importlib
from functools import cache, partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
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
        specs = {"n_samples": len(self.dataset)}  # type: ignore
        run_cond = hasattr(self.dataset, "get_data_specs")
        if run_cond and isinstance(self.dataset.get_data_specs(), dict):  # type: ignore
            user_specs = self.dataset.get_data_specs()  # type: ignore
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
        data_item_info = get_data_item_info(self.dataset[0])
        my_collate = self.get_custom_collate(data_item_info)  # type: ignore
        yield from DataLoader(
            dataset=self.dataset,
            batch_sampler=batch_sampler,
            collate_fn=my_collate,
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
                    "All keys of the dictionnary returned by your original Torch"
                    "Dataset method `get_specs()` must conform to one of the fields"
                    f"found in `declearn.dataset.DataSpecs`. '{key}' did not. "
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


def get_data_item_info(data_item) -> Optional[Dict[str, Any]]:
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
    if data_item_info["type"] == torch.Tensor:
        out = (original_batch, None, None)
    else:
        patch = [None for _ in range(3 - data_item_info["len"])]
        out = (*original_batch, *patch)  # type: ignore
    return out
