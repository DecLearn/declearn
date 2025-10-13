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

"""Utils to save and load array-like data to and from various file formats."""

import functools
import os
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix  # type: ignore
from sklearn.datasets import load_svmlight_file  # type: ignore

from declearn.dataset.utils._sparse import sparse_from_file, sparse_to_file
from declearn.typing import DataArray

__all__ = [
    "load_data_array",
    "save_data_array",
]


def load_data_array(
    path: str,
    **kwargs: Any,
) -> DataArray:
    """Load a data array from a dump file.

    Supported file extensions
    -------------------------
    - `.csv`:
        csv file, comma-delimited by default.
        Any keyword arguments to `pandas.read_csv` may be passed.
    - `.npy`:
        Non-pickle numpy array dump, created with `numpy.save`.
    - `.sparse`:
        Scipy sparse matrix dump, created with the custom
        `declearn.data.sparse.sparse_to_file` function.
    - `.svmlight`:
        SVMlight sparse matrix and labels array dump.
        Parse using `sklearn.load_svmlight_file`, and
        return either features or labels based on the
        `which` int keyword argument (default: 0, for
        features).

    Parameters
    ----------
    path: str
        Path to the data array dump file.
        Extension must be adequate to enable proper parsing;
        see list of supported extensions above.
    **kwargs:
        Extension-type-based keyword parameters may be passed.
        See above for details.

    Returns
    -------
    data: numpy.ndarray or pandas.DataFrame or scipy.spmatrix
        Reloaded data array.

    Raises
    ------
    TypeError
        If `path` is of unsupported extension.

    Any exception raised by data-loading functions may also be raised
    (e.g. if the file cannot be proprely parsed).
    """
    ext = os.path.splitext(path)[1]
    if ext == ".csv":
        return pd.read_csv(path, **kwargs)
    if ext == ".npy":
        return np.load(path, allow_pickle=False)
    if ext == ".sparse":
        return sparse_from_file(path)
    if ext == ".svmlight":
        which = kwargs.get("which", 0)
        return load_svmlight_file(path)[which]
    raise TypeError(f"Unsupported data array file extension: '{ext}'.")


def save_data_array(
    path: str,
    array: Union[DataArray, pd.Series],
) -> str:
    """Save a data array to a dump file.

    Supported types of data arrays
    ------------------------------
    - `pandas.DataFrame` or `pandas.Series`:
        Dump to a comma-separated `.csv` file.
    - `numpy.ndarray`:
        Dump to a non-pickle `.npy` file.
    - `scipy.sparse.spmatrix`:
        Dump to a `.sparse` file, using a custom format
        and `declearn.data.sparse.sparse_to_file`.

    Parameters
    ----------
    path: str
        Path to the file where to dump the array.
        Appropriate file extension will be added when
        not present (i.e. `path` may be a basename).
    array: data array structure (see above)
        Data array that needs dumping to file.
        See above for supported types and associated
        behaviours.

    Returns
    -------
    path: str
        Path to the created file dump, based on the input
        `path` and the chosen file extension (see above).

    Raises
    ------
    TypeError
        If `array` is of unsupported type.
    """
    # Select a file extension and set up the array-dumping function.
    if isinstance(array, (pd.DataFrame, pd.Series)):
        ext = ".csv"
        save = functools.partial(
            array.to_csv, sep=",", encoding="utf-8", index=False
        )
    elif isinstance(array, np.ndarray):
        ext = ".npy"
        save = functools.partial(np.save, arr=array)
    elif isinstance(array, spmatrix):
        ext = ".sparse"
        save = functools.partial(sparse_to_file, matrix=array)
    else:
        raise TypeError(f"Unsupported data array type: '{type(array)}'.")
    # Ensure proper naming. Save the array. Return the path.
    if not path.endswith(ext):
        path += ext
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save(path)
    return path
