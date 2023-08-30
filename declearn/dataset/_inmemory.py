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

"""Dataset implementation to serve scikit-learn compatible in-memory data."""

import os
import warnings
from typing import Any, Dict, Iterator, List, Optional, Set, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix  # type: ignore
from sklearn.datasets import load_svmlight_file  # type: ignore
from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.dataset._base import Dataset, DataSpecs
from declearn.dataset.utils import load_data_array, save_data_array
from declearn.typing import Batch
from declearn.utils import json_dump, json_load, register_type

__all__ = [
    "InMemoryDataset",
]


DataArray = Union[np.ndarray, pd.DataFrame, spmatrix]


@register_type(group="Dataset")
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
    into two distinct base classes to articulate back into this one.

    Attributes
    ----------
    data: data array
        Data array containing samples, with all input features
        (and optionally more columns).
    target: data array or None
        Optional data array containing target labels ~ values.
    f_cols: list[int] or list[str] or None
        Optional subset of `data` columns to restrict yielded
        input features (i.e. batches' first array) to which.
    """

    # attributes serve clarity; pylint: disable=too-many-instance-attributes
    # arguments serve modularity; pylint: disable=too-many-arguments

    def __init__(
        self,
        data: Union[DataArray, str],
        target: Optional[Union[DataArray, str]] = None,
        s_wght: Optional[Union[DataArray, str]] = None,
        f_cols: Optional[Union[List[int], List[str]]] = None,
        expose_classes: bool = False,
        expose_data_type: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Instantiate the dataset interface.

        We thereafter use the term "data array" to designate
        an instance that is either a numpy ndarray, a pandas
        DataFrame or a scipy spmatrix.

        See the `load_data_array` function in dataset._utils for details
        on supported file formats.

        Parameters
        ----------
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
        expose_classes: bool, default=False
            Whether the dataset should be used for classification, in
            which case the unique values of `target` are exposed under
            `self.classes` and exported by `self.get_data_specs()`).
        expose_data_type: bool, default=False
            Whether the dataset should expose the data type , in
            which case check if the type is unique and exposed it
            under `self.data_type` and exported by `self.get_data_specs()`).
        seed: int or None, default=None
            Optional seed for the random number generator based on which
            the dataset is (optionally) shuffled when generating batches.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        self._data_path = None  # type: Optional[str]
        self._trgt_path = None  # type: Optional[str]
        # Assign the main data array.
        if isinstance(data, str):
            self._data_path = data
            data = load_data_array(data)
        self.data = data
        # Assign the optional input features list.
        self.f_cols = f_cols
        # Assign the (optional) target data array.
        if isinstance(target, str):
            self._trgt_path = target
            if (
                isinstance(self.data, pd.DataFrame)
                and target in self.data.columns
            ):
                if f_cols is None:
                    self.f_cols = self.f_cols or list(self.data.columns)
                    self.f_cols.remove(target)  # type: ignore
                target = self.data[target]
            else:
                target = load_data_array(target)
                if (
                    isinstance(target, pd.DataFrame)
                    and len(target.columns) == 1
                ):
                    target = target.iloc[:, 0]
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
                s_wght = load_data_array(s_wght)
        self.weights = s_wght
        # Assign the 'expose_classes' attribute.
        self.expose_classes = expose_classes
        self.expose_data_type = expose_data_type
        # Assign a random number generator.
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def feats(
        self,
    ) -> DataArray:
        """Input features array."""
        if self.f_cols is None:
            return self.data
        if isinstance(self.data, pd.DataFrame):
            if isinstance(self.f_cols[-1], str):
                return self.data.loc[:, self.f_cols]  # type: ignore
            return self.data.iloc[:, self.f_cols]  # type: ignore
        return self.data[:, self.f_cols]  # type: ignore

    @property
    def classes(self) -> Optional[Set[Any]]:
        """Unique target classes."""
        if (not self.expose_classes) or (self.target is None):
            return None
        if isinstance(self.target, pd.DataFrame):
            c_list = self.target.unstack().unique().tolist()  # type: ignore
            return set(c_list)
        if isinstance(self.target, pd.Series):
            return set(self.target.unique().tolist())
        if isinstance(self.target, np.ndarray):
            return set(np.unique(self.target).tolist())
        if isinstance(self.target, spmatrix):
            return set(np.unique(self.target.tocsr().data).tolist())
        raise TypeError(  # pragma: no cover
            f"Invalid 'target' attribute type: '{type(self.target)}'."
        )

    @property
    def data_type(self) -> Optional[str]:
        """Unique data type."""
        if not self.expose_data_type:
            return None
        if isinstance(self.feats, pd.DataFrame):
            dtypes = {str(t) for t in list(self.feats.dtypes)}
            if len(dtypes) > 1:
                raise ValueError(
                    "Cannot work with mixed data types:"
                    "ensure the `data` attribute has unique dtype"
                )
            return list(dtypes)[0]
        if isinstance(self.feats, (pd.Series, np.ndarray, spmatrix)):
            return str(self.feats.dtype)
        raise TypeError(  # pragma: no cover
            f"Invalid 'data' attribute type: '{type(self.target)}'."
        )

    @staticmethod
    def load_data_array(
        path: str,
        **kwargs: Any,
    ) -> DataArray:  # pragma: no cover
        """Load a data array from a dump file.

        As of declearn v2.2, this staticmethod is DEPRECATED in favor of
        `declearn.dataset.utils.load_data_array`, which is now calls. It
        will be removed in v2.4 and/or v3.0.

        See [declearn.dataset.utils.load_data_array][] for more details.
        """
        warnings.warn(
            "'InMemoryDataset.load_data_array' has been deprecated in favor"
            " of `declearn.dataset.utils.load_data_array`. It will be removed"
            " in version 2.4 and/or 3.0.",
            category=DeprecationWarning,
        )
        return load_data_array(path, **kwargs)

    @staticmethod
    def save_data_array(
        path: str,
        array: Union[DataArray, pd.Series],
    ) -> str:  # pragma: no cover
        """Save a data array to a dump file.

        As of declearn v2.2, this staticmethod is DEPRECATED in favor of
        `declearn.dataset.utils.save_data_array`, which is now calls. It
        will be removed in v2.4 and/or v3.0.

        See [declearn.dataset.utils.save_data_array][] for more details.
        """
        warnings.warn(
            "'InMemoryDataset.save_data_array' has been deprecated in favor"
            " of `declearn.dataset.utils.save_data_array`. It will be removed"
            " in version 2.4 and/or 3.0.",
            category=DeprecationWarning,
        )
        return save_data_array(path, array)

    @classmethod
    def from_svmlight(
        cls,
        path: str,
        f_cols: Optional[List[int]] = None,
        dtype: Union[str, np.dtype] = "float64",
    ) -> Self:
        """Instantiate a InMemoryDataset from a svmlight file.

        A SVMlight file contains both input features (as a sparse
        matrix in CSR format) and target labels. This method uses
        `sklearn.datasets.load_svmlight_file` to parse this file.

        Parameters
        ----------
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

        Parameters
        ----------
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
        info["type"] = "InMemoryDataset"  # NOTE: for backward compatibility
        # Optionally create data dumps. Record data dumps' paths.
        # fmt: off
        info["data"] = (
            self._data_path or
            save_data_array(os.path.join(folder, "data"), self.data)
        )
        info["target"] = None if self.target is None else (
            self._trgt_path or
            save_data_array(os.path.join(folder, "trgt"), self.target)
        )
        info["s_wght"] = None if self.weights is None else (
            self._wght_path or
            save_data_array(os.path.join(folder, "wght"), self.weights)
        )
        # fmt: on
        info["f_cols"] = self.f_cols
        info["expose_classes"] = self.expose_classes
        info["seed"] = self.seed
        # Write the information to the JSON file.
        dump = {"name": self.__class__.__name__, "config": info}
        json_dump(dump, path, indent=2)

    @classmethod
    def load_from_json(
        cls,
        path: str,
    ) -> Self:
        """Instantiate a dataset based on local files.

        Parameters
        ----------
        path: str
            Path to the main JSON file where to dump the dataset.
            Additional files may be created in the same folder.
        """
        # Read and parse the JSON file and check its specs conformity.
        dump = json_load(path)
        if "config" not in dump:
            raise KeyError("Missing key in the JSON file: 'config'.")
        info = dump["config"]
        for key in ("data", "target", "s_wght", "f_cols"):
            if key not in info:
                error = f"Missing key in the JSON file: 'config/{key}'."
                raise KeyError(error)
        info.pop("type", None)
        # Instantiate the object and return it.
        return cls(**info)

    def get_data_specs(
        self,
    ) -> DataSpecs:
        """Return a DataSpecs object describing this dataset."""
        return DataSpecs(
            n_samples=self.feats.shape[0],
            features_shape=self.feats.shape[1:],
            classes=self.classes,
            data_type=self.data_type,
        )

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
        inputs: data array
            Input features; scikit-learn's `X`.
        targets: data array or None
            Optional target labels or values; scikit-learn's `y`.
        weights: data array or None
            Optional sample weights; scikit-learn's `sample_weight`.

        Notes
        -----
        - In this context, a 'data array' is either a numpy array,
          scipy sparse matrix, pandas dataframe or pandas series.
        - Batched arrays are aligned along their first axis.
        """
        if poisson:
            order = self._poisson_sampling(batch_size, drop_remainder)
            # Enable slicing of the produced boolean mask in `_build_iterator`.
            batch_size = order.shape[1]  # n_samples
            order = order.flatten()
        else:
            order = self._samples_batching(
                batch_size, shuffle, replacement, drop_remainder
            )
        # Build array-wise batch iterators.
        iterators = [
            self._build_iterator(data, batch_size, order)
            for data in (self.feats, self.target, self.weights)
        ]
        # Yield tuples zipping the former.
        yield from zip(*iterators)

    def _samples_batching(
        self,
        batch_size: int,
        shuffle: bool = False,
        replacement: bool = False,
        drop_remainder: bool = True,
    ) -> np.ndarray:
        """Create an ordering of samples to conduct their batching.

        Parameters
        ----------
        batch_size: int
            Number of samples per batch.
        shuffle: bool, default=False
            Whether to shuffle data samples prior to batching.
            Note that the shuffling will differ on each call
            to this method.
        replacement: bool, default=False
            Whether to draw samples with replacement.
            Unused if `shuffle=False`.
        drop_remainder: bool, default=True
            Whether to drop the last batch if it contains less
            samples than `batch_size`, or yield it anyway.

        Returns
        -------
        order: 1-d numpy.ndarray
            Array indexing the raw samples for their batching.
            The `_build_iterator` method may be used to slice
            through this array to extract batches from the raw
            data arrays.
        """
        order = np.arange(self.feats.shape[0])
        # Optionally set up samples' shuffling.
        if shuffle:
            if replacement:
                order = self._rng.choice(order, size=len(order), replace=True)
            else:
                order = self._rng.permutation(order)
        # Optionally drop last batch if its size is too small.
        if drop_remainder:
            limit = len(order) - (len(order) % batch_size)
            order = order[:limit]
            if len(order) == 0:
                raise ValueError(
                    "The dataset is smaller than `batch_size`, so that "
                    "`drop_remainder=True` results in an empty iterator."
                )
        # Return the ordering.
        return order

    def _poisson_sampling(
        self,
        batch_size: int,
        drop_remainder: bool = True,
    ) -> np.ndarray:
        """Create a boolean masking of samples to conduct their batching.

        Poisson sampling consists in making up batches by drawing from a
        bernoulli distribution for each and every sample in the dataset,
        to decide whether it should be included in the batch. As a result
        batches vary in size, and a sample may appear zero or multiple
        times in the set of batches drawn for a (pseudo-)epoch.

        Parameters
        ----------
        batch_size: int
            Desired average number of samples per batch.
            The sample rate for the Poisson sampling procedure
            is set to `batch_size / n_samples`.
        drop_remainder: bool, default=True
            Since Poisson sampling does not result in fixed-size
            batches, this parameter is interpreted as whether to
            set the number of batches to `floor(1 / sample_rate)`
            rather than `ceil(1 / sample_rate)`.

        Returns
        -------
        bmask: 2-d numpy.ndarray
            Array with shape `(n_batches, n_samples)`, each row
            of which provides with a boolean mask that should be
            used to produce a batch from the raw data samples.
        """
        # Compute the desired sample rate and number of batches.
        n_samples = self.feats.shape[0]
        sample_rate = batch_size / n_samples
        n_batches = n_samples // batch_size
        if (n_samples % batch_size) and not drop_remainder:
            n_batches += 1
        # Conduct Poisson sampling of all batches.
        bmask = self._rng.uniform(size=(n_batches, n_samples)) < sample_rate
        return bmask

    def _build_iterator(
        self,
        data: Optional[DataArray],
        batch_size: int,
        order: np.ndarray,
    ) -> Iterator[Optional[ArrayLike]]:
        """Yield batches extracted from a data array.

        Parameters
        ----------
        data: optional data array
            Data from which to derive the yielded batches.
            If None, yield None as many times as `order` specifies it.
        batch_size: int
            Number of samples to include per batch.
        order: np.ndarray
            Array containing a pre-computed samples' ordering.
            Yield batches of samples drawn in that order from `data`.

        Yields
        ------
        batch: optional data array
            Slice of `data`, or None if `data` is None.
        """
        if data is None:
            yield from (None for _ in range(0, len(order), batch_size))
        else:
            # Ensure slicing compatibility for pandas structures.
            if isinstance(data, (pd.DataFrame, pd.Series)):
                data = data.values
            # Iteratively yield slices of the data array.
            for idx in range(0, len(order), batch_size):
                end = idx + batch_size
                yield data[order[idx:end]]
