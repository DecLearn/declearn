# coding: utf-8

"""Dataset abstraction API."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Set

from numpy.typing import ArrayLike


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
        ) -> Iterator[List[Optional[ArrayLike]]]:
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
        batch: list of (optional) array-like elements
            Depending on the actual dataset used (and the
            learning task at hand), a batch may contain a
            varying-number of elements; e.g. input features,
            target labels, sample weights...
        """
        return NotImplemented
