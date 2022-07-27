# coding: utf-8

"""Dataset implementation to serve scikit-learn compatible in-memory data."""

import functools
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Set, Union

import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix  # type: ignore
from sklearn.datasets import load_svmlight_file  # type: ignore

from declearn2.dataset._base import Dataset, DataSpecs
from declearn2.dataset._sparse import sparse_from_file, sparse_to_file


__all__ = [
    'SklearnDataset',
]


DataArray = Union[np.ndarray, pd.DataFrame, spmatrix]


class SklearnDataset(Dataset):
    """Dataset subclass serving numpy(-like) memory-loaded data arrays.

    This subclass implements:
    * yielding (X, [y]) batches matching the scikit-learn API,
      with data wrapped as numpy arrays, scipy sparse matrix
      or pandas dataframes (or series)
    * loading the source data from which batches are derived
      fully in memory, with support for some standard file
      formats

    Note: future code refactoring may divide these functionalities
          into two distinct base classes to articulate back into
          this one.

    Attributes:
    ----------
    data: data array
        Data array containing samples, with all input features
        (and optionally more columns).
    target: data array or None
        Optional data array containing target labels.
    f_cols: list[int], list[str] or None
        Optional subset of `data` columns to restrict yielded
        input features (i.e. batches' first array) to which.
    """

    _type_key = "SklearnDataset"

    def __init__(
            self,
            data: Union[DataArray, str],
            target: Optional[Union[DataArray, str]] = None,
            f_cols: Optional[Union[List[int], List[str]]] = None,
        ) -> None:
        """Instantiate the dataset interface.

        We thereafter use the term "data array" to designate
        an instance that is either a numpy ndarray, a pandas
        DataFrame or a scipy spmatrix.

        See the `load_data_array` method for details
        on supported file formats.

        Arguments:
        ---------
        data: data array or str
            Main data array which contains input features (and possibly
            more), or path to a dump file from which it is to be loaded.
        target: data array or str or None, default=None
            Optional data array containing target labels (for supervised
            learning), or path to a dump file from which to load it.
            If `data` is a pandas DataFrame (or a path to a csv file),
            `target` may be the name of a column to use as labels (and
            thus not to use as input feature unless listed in `f_cols`).
        f_cols: list[int] or list[str] or None, default=None
            Optional list of columns in `data` to use as input features
            (other columns will not be included in the first array of
            batches yielded by `self.generate_batches(...)`).
        """
        self._data_path = None  # type: Optional[str]
        self._trgt_path = None  # type: Optional[str]
        # Assign the main data array.
        if isinstance(data, str):
            self._data_path = data
            data = self.load_data_array(data)
        self.data = data
        # Assign the target data array.
        if isinstance(target, str):
            self._trgt_path = target
            if isinstance(self.data, pd.DataFrame):
                try:
                    target = self.data[target]
                except KeyError:
                    target = self.load_data_array(target)
                else:
                    if f_cols is None:
                        f_cols = [c for c in self.data.columns if c != target]
            else:
                target = self.load_data_array(target)
        self.target = target
        # Assign the optional input features list.
        self.f_cols = f_cols

    @property
    def feats(
            self,
        ) -> DataArray:
        """Input features array."""
        if self.f_cols is None:
            return self.data
        if isinstance(self.data, pd.DataFrame):
            if isinstance(self.f_cols[-1], str):
                return self.data.loc[:, self.f_cols]
            return self.data.iloc[:, self.f_cols]
        return self.data[:, self.f_cols]  # type: ignore

    @property
    def classes(
            self
        ) -> Optional[Set[Any]]:
        """Unique target classes."""
        if self.target is None:
            return None
        if isinstance(self.target, pd.DataFrame):
            return set(self.target.unstack().unique())
        if isinstance(self.target, pd.Series):
            return set(self.target.unique())
        if isinstance(self.target, np.ndarray):
            return set(np.unique(self.target))
        if isinstance(self.target, spmatrix):
            return set(np.unique(self.target.tocsr().data))
        raise TypeError(
            f"Invalid 'target' attribute type: '{type(self.target)}'."
        )

    @property
    def _batchable_arrays(
            self
        ) -> List[Optional[DataArray]]:
        """List of (optional) arrays to yield from in `generate_batches`.

        This attribute is meant for private use only, for the notable
        purpose of being overridden in children classes in order to
        add (or remove) information fields from yielded batches, e.g.
        to add sample-wise weights.
        """
        return [self.feats, self.target]

    @staticmethod
    def load_data_array(
            path: str,
            **kwargs: Any,
        ) -> DataArray:
        """Load a data array from a dump file.

        Supported file extensions are:
        .csv:
            csv file, comma-delimited by default.
            Any keyword arguments to `pandas.read_csv` may be passed.
        .npy:
            Non-pickle numpy array dump, created with `numpy.save`.
        .sparse:
            Scipy sparse matrix dump, created with the custom
            `declearn.data.sparse.sparse_to_file` function.
        .svmlight:
            SVMlight sparse matrix and labels array dump.
            Parse using `sklearn.load_svmlight_file`, and
            return either features or labels based on the
            `which` int keyword argument (default: 0, for
            features).

        Arguments:
        ---------
        path: str
            Path to the data array dump file.
            Extension must be adequate to enable proper parsing;
            see list of supported extensions above.
        **kwargs:
            Extension-type-based keyword parameters may be passed.
            See above for details.

        Returns:
        -------
        data: numpy.ndarray or pandas.DataFrame or scipy.spmatrix
            Reloaded data array.

        Raises:
        ------
        TypeError:
            If `path` is of unsupported extension.
        Any exception raised by data-loading functions may also be
        raised (e.g. if the file cannot be proprely parsed).
        """
        ext = os.path.splitext(path)[1]
        if ext == ".csv":
            return pd.read_csv(path, **kwargs).values
        if ext == ".npy":
            return np.load(path, allow_pickle=False)
        if ext == ".sparse":
            return sparse_from_file(path)
        if ext == ".svmlight":
            which = kwargs.get("which", 0)
            return load_svmlight_file(path)[which]
        raise TypeError(f"Unsupported data array file extension: '{ext}'.")

    @staticmethod
    def save_data_array(
            path: str,
            array: Union[DataArray, pd.Series],
        ) -> str:
        """Save a data array to a dump file.

        Supported types of data arrays are:
        pandas.DataFrame or pandas.Series:
            Dump to a comma-separated ".csv" file.
        numpy.ndarray:
            Dump to a non-pickle ".npy" file.
        scipy.sparse.spmatrix:
            Dump to a ".sparse" file, using a custom format
            and `declearn.data.sparse.sparse_to_file`.

        Arguments:
        ---------
        path: str
            Path to the file where to dump the array.
            Appropriate file extension will be added when
            not present (i.e. `path` may be a basename).
        array: data array structure (see above)
            Data array that needs dumping to file.
            See above for supported types and associated
            behaviours.

        Returns:
        -------
        path: str
            Path to the created file dump, based on the input
            `path` and the chosen file extension (see above).

        Raises:
        ------
        TypeError:
            If `array` is of unsupported type.
        """
        # Select a file extension and set up the array-dumping function.
        if isinstance(array, (pd.DataFrame, pd.Series)):
            ext = ".csv"
            save = functools.partial(array.to_csv, sep=",", encoding="utf-8")
        elif isinstance(array, np.ndarray):
            ext = ".npy"
            save = functools.partial(np.save, arr=array)
        elif isinstance(array, spmatrix):
            ext = ".sparse"
            save = functools.partial(sparse_to_file, array=array)
        else:
            raise TypeError(f"Unsupported data array type: '{type(array)}'.")
        # Ensure proper naming. Save the array. Return the path.
        if not path.endswith(ext):
            path += ext
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save(path)
        return path

    @classmethod
    def from_svmlight(
            cls,
            path: str,
            f_cols: Optional[List[int]] = None,
            dtype: Union[str, np.dtype] = 'float64',
        ) -> "SklearnDataset":
        """Instantiate a SklearnDataset from a svmlight file.

        A SVMlight file contains both input features (as a sparse
        matrix in CSR format) and target labels. This method uses
        `sklearn.datasets.load_svmlight_file` to parse this file.

        Arguments:
        ---------
        path: str
            Path to the SVMlight file from which to load the `data`
            and `target` parameters used to isinstantiate.
        f_cols: list[int] or None, default=None
            Optional list of columns of the loaded sparse matrix
            to restrict yielded input features to which.
        dtype: str or numpy.dtype, default='float64'
            Dtype of the reloaded input features' matrix.
        """
        # false-positive warning; pylint: disable=unbalanced-tuple-unpacking
        data, target = load_svmlight_file(path, dtype=dtype)
        return cls(data=data, target=target, f_cols=f_cols)

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

        Note: In case created (non-JSON) data files are moved,
              the paths documented in the JSON file will need
              to be updated.
        """
        path = os.path.abspath(path)
        folder = os.path.dirname(path)
        info = {}  # type: Dict[str, Any]
        info["type"] = self._type_key
        # Optionally create data dumps. Record data dumps' paths.
        info["data"] = (
            self._data_path
            or self.save_data_array(os.path.join(folder, "data"), self.data)
        )
        info["target"] = None if self.target is None else (
            self._trgt_path
            or self.save_data_array(os.path.join(folder, "trgt"), self.target)
        )
        info["f_cols"] = self.f_cols
        # Write the information to the JSON file.
        with open(path, "w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)

    @classmethod
    def load_from_json(
            cls,
            path: str,
        ) -> 'SklearnDataset':
        """Instantiate a dataset based on local files.

        Arguments:
        ---------
        path: str
            Path to the main JSON file where to dump the dataset.
            Additional files may be created in the same folder.
        """
        # Read and parse the JSON file and check its specs conformity.
        with open(path, "r", encoding="utf-8") as file:
            info = json.load(file)
        for key in ("type", "data", "target", "f_cols"):
            if key not in info:
                raise KeyError(f"Missing key in the JSON file: '{key}'.")
        key = info.pop("type")
        if key != cls._type_key:
            raise TypeError(
                f"Incorrect 'type' field: got '{key}', "\
                f"expected '{cls._type_key}'."
            )
        # Instantiate the object and return it.
        return cls(**info)

    def get_data_specs(
            self,
        ) -> DataSpecs:
        """Return a DataSpecs object describing this dataset."""
        return DataSpecs(
            n_samples=self.feats.shape[0],
            n_features=self.feats.shape[1],
            classes=self.classes
        )

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
            Whether to shuffle data samples prior to batching it.
            If True, use `rng` to generate samples' permutation,
            or call `numpy.random.default_rng()` if not set.
        seed: int or None, default=None
            Optional seed to the random-numbers generator
            used to shuffle samples prior to batching.
            Only used when `shuffle=True`.
        drop_remainder: bool, default=True
            Whether to drop the last batch if it contains less
            samples than `batch_size`, or yield it anyway.

        Yields:
        ------
        batch: 2-elements list
            List of (optional) array-like elements, aligned along
            their first axis. Here, `batch` is [X, y] with:
            * X: array-like
                Batch of input features, as a 2-D array.
                This may be a numpy array, scipy sparse matrix
                or pandas dataframe depending on `self.data`.
            * y: array-like or None
                Batch of labels associated with the features.
                If `self.target` is None, yield None values
                (e.g. for unsupervised learning models.).
                It may otherwise be a 1-D or 2-D numpy array,
                scipy sparse matrix, pandas dataframe or pandas
                series, depending on `self.target`.
        """
        # Optionally set up samples' shuffling.
        if shuffle:
            rng = np.random.default_rng(seed)
            order = rng.permutation(self.feats.shape[0])
        else:
            order = np.arange(self.feats.shape[0])
        # Optionally drop last batch if its size is too small.
        if drop_remainder:
            order = order[:len(order) - (len(order) % batch_size)]
        # Build array-wise batch iterators.
        iterators = [
            self._build_iterator(data, batch_size, order)
            for data in self._batchable_arrays
        ]
        # Yield tuples zipping the former.
        for batch in zip(*iterators):
            yield list(batch)

    def _build_iterator(
            self,
            data: Optional[DataArray],
            batch_size: int,
            order: np.ndarray,
        ) -> Iterator[Optional[ArrayLike]]:
        """Yield batches extracted from a data array.

        data: optional data array
            Data from which to derive the yielded batches.
            If None, yield None as many times as `order` specifies it.
        batch_size: int
            Number of samples to include per batch.
        order: np.ndarray
            Array containing a pre-computed samples' ordering.
            Yield batches of samples drawn in that order from `data`.

        Yield slices of `data`, or None values if `data` is None.
        """
        if data is None:
            yield from (None for _ in range(0, len(order), batch_size))
        else:
            # Ensure slicing compatibility for pandas structures.
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data = data.iloc
            # Iteratively yield slices of the data array.
            for idx in range(0, len(order), batch_size):
                yield data[order[idx:idx+batch_size]]