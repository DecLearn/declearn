# coding: utf-8

"""Dataset abstraction API."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Set

from declearn.typing import Batch
from declearn.utils import access_registered, create_types_registry, json_load


__all__ = [
    "DataSpecs",
    "Dataset",
    "load_dataset_from_json",
]


@dataclass
class DataSpecs:
    """Dataclass to wrap a dataset's metadata."""

    n_samples: int
    n_features: int
    classes: Optional[Set[Any]] = None


@create_types_registry
class Dataset(metaclass=ABCMeta):
    """Abstract class defining an API to access training or testing data.

    A 'Dataset' is an interface towards data that exposes methods
    to query batched data samples and key metadata while remaining
    agnostic of the way the data is actually being loaded (from a
    source file, a database, a network reader, another API...).

    This is notably done to allow clients to use distinct data
    storage and loading architectures, even implementing their
    own subclass if needed, while ensuring that data access is
    straightforward to specify as part of FL algorithms.
    """

    _type_key: str = NotImplemented

    @abstractmethod
    def save_to_json(
        self,
        path: str,
    ) -> None:
        """Write a JSON file enabling dataset re-creation.

        Parameters
        ----------
        path: str
            Path to the main JSON file where to dump the dataset.
            Additional files may be created in the same folder.
        """
        return None

    @classmethod
    @abstractmethod
    def load_from_json(
        cls,
        path: str,
    ) -> "Dataset":
        """Instantiate a dataset based on local files."""
        return NotImplemented

    @abstractmethod
    def get_data_specs(
        self,
    ) -> DataSpecs:
        """Return a DataSpecs object describing this dataset."""
        return NotImplemented

    @abstractmethod
    def generate_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_remainder: bool = True,
        poisson: bool = False,
    ) -> Iterator[Batch]:
        """Yield batches of data samples.

        Parameters
        ----------
        batch_size: int
            Number of samples per batch.
        shuffle: bool, default=False
            Whether to shuffle data samples prior to batching.
            Note that the shuffling will differ on each call
            to this method.
        drop_remainder: bool, default=True
            Whether to drop the last batch if it contains less
            samples than `batch_size`, or yield it anyway.
            If `poisson=True`, this is used to determine the number
            of returned batches (notwithstanding their actual size).
        poisson: bool, default=False
            Whether to use Poisson sampling, i.e. make up batches by
            drawing samples with replacement, resulting in variable-
            size batches and samples possibly appearing in zero or in
            multiple emitted batches (but at most once per batch).
            Useful to maintain tight Differential Privacy guarantees.

        Yields
        ------
        inputs: (2+)-dimensional data array or list of data arrays
            Input features of that batch.
        targets: data array, list of data arrays or None
            Target labels or values of that batch.
            May be None for unsupervised or semi-supervised tasks.
        weights: 1-d data array or None
            Optional weights associated with the samples, that are
            typically used to balance a model's loss or metrics.
        """
        return NotImplemented


def load_dataset_from_json(path: str) -> Dataset:
    """Instantiate a dataset based on a JSON dump file.

    Parameters
    ----------
    path: str
        Path to a JSON file output by the `save_to_json`
        method of the Dataset that is being reloaded.
        The actual type of dataset should be specified
        under the "name" field of that file.

    Returns
    -------
    dataset: Dataset
        Dataset (subclass) instance, reloaded from JSON.
    """
    dump = json_load(path)
    cls = access_registered(dump["name"], group="Dataset")
    return cls.load_from_json(path)  # type: ignore
