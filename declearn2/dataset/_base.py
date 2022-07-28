# coding: utf-8

"""Dataset abstraction API."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Set

from declearn2.typing import Batch


__all__ = [
    'DataSpecs',
    'Dataset',
]


@dataclass
class DataSpecs:
    """Dataclass to wrap a dataset's metadata."""
    n_samples: int
    n_features: int
    classes: Optional[Set[Any]] = None


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

        Arguments:
        ---------
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
        ) -> 'Dataset':
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
            seed: Optional[int] = None,
            drop_remainder: bool = True,
        ) -> Iterator[Batch]:
        """Yield batches of data samples.

        Arguments:
        ---------
        batch_size: int
            Number of samples per batch.
        shuffle: bool, default=False
            Whether to shuffle data samples prior to batching.
        seed: int or None, default=None
            Optional seed to the random-numbers generator
            used to generate batches (e.g. for shuffling).
        drop_remainder: bool, default=True
            Whether to drop the last batch if it contains less
            samples than `batch_size`, or yield it anyway.

        Yields:
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
