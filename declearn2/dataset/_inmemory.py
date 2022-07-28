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
from declearn2.typing import Batch


__all__ = [
    'InMemoryDataset',
]


DataArray = Union[np.ndarray, pd.DataFrame, spmatrix]


class InMemoryDataset(Dataset):
    """Dataset subclass serving numpy(-like) memory-loaded data arrays.

    This subclass implements:
    * yielding (X, [y], [w]) batches matching the scikit-learn API,
      with data wrapped as numpy arrays, scipy sparse matrices,
      or pandas dataframes (or series for y and w)
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
    f_cols: list[int] or list[str] or None
        Optional subset of `data` columns to restrict yielded
        input features (i.e. batches' first array) to which.
    """

    _type_key = "InMemoryDataset"

    def __init__(
            self,
            data: Union[DataArray, str],
            target: Optional[Union[DataArray, str]] = None,
            s_wght: Optional[Union[DataArray, str]] = None,
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
        s_wght: int or str or function or None, default=None
            Optional data array containing sample weights, or path to a
            dump file from which to load it.
            If `data` is a pandas DataFrame (or a path to a csv file),
            `s_wght` may be the name of a column to use as labels (and
            thus not to use as input feature unless listed in `f_cols`).
        f_cols: list[int] or list[str] or None, default=None
            Optional list of columns in `data` to use as input features
            (other columns will not be included in the first array of
            the batches yielded by `self.generate_batches(...)`).
        """
        self._data_path = None  # type: Optional[str]
        self._trgt_path = None  # type: Optional[str]
        # Assign the main data array.
        if isinstance(data, str):
            self._data_path = data
            data = self.load_data_array(data)
        self.data = data
        # Assign the optional input features list.
        self.f_cols = f_cols
        # Assign the (optional) target data array.
        if isinstance(target, str):
            self._trgt_path = target
            if isinstance(self.data, pd.DataFrame):
                if target in self.data.columns:
                    if f_cols is None:
                        self.f_cols = self.f_cols or list(self.data.columns)
                        self.f_cols.remove(target)  # type: ignore
                    target = self.data[target]
            else:
                target = self.load_data_array(target)
        self.target = target
        # Assign the (optional) sample weights data array.
        if isinstance(s_wght, str):
            self._wght_path = s_wght
            if isinstance(self.data, pd.DataFrame):
                if s_wght in self.data.columns:
                    if f_cols is None:
                        self.f_cols = self.f_cols or list(self.data.columns)
                        self.f_cols.remove(s_wght)  # type: ignore
                    s_wght = self.data[s_wght]
            else:
                s_wght = self.load_data_array(s_wght)
        self.weights = s_wght

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
        ) -> "InMemoryDataset":
        """Instantiate a InMemoryDataset from a svmlight file.

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
        info["s_wght"] = None if self.weights is None else (
            self._wght_path
            or self.save_data_array(os.path.join(folder, "wght"), self.weights)
        )
        info["f_cols"] = self.f_cols
        # Write the information to the JSON file.
        with open(path, "w", encoding="utf-8") as file:
            json.dump(info, file, indent=2)

    @classmethod
    def load_from_json(
            cls,
            path: str,
        ) -> 'InMemoryDataset':
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
        for key in ("type", "data", "target", "s_wght", "f_cols"):
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
        ) -> Iterator[Batch]:
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
        inputs: data array
            Input features; scikit-learn's `X`.
        targets: data array or None
            Optional target labels or values; scikit-learn's `y`.
        weights: data array or None
            Optional sample weights; scikit-learn's `sample_weight`.

        Note: in this context, a 'data array' is either a numpy array,
              scipy sparse matrix, pandas dataframe or pandas series.
        Note: batched arrays are aligned along their first axis.
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
            for data in (self.feats, self.target, self.weights)
        ]
        # Yield tuples zipping the former.
        yield from zip(*iterators)

    def _build_iterator(
            self,
            data: Optional[DataArray],
            batch_size: int,
            order: np.ndarray,
        ) -> Iterator[Optional[ArrayLike]]:
        """Yield batches extracted from a data array.

        Arguments:
        ---------
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
