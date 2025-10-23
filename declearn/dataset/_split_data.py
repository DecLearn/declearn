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

"""Script to split data into heterogeneous shards and save them.

Available splitting scheme:

* "iid", split the dataset through iid random sampling.
* "labels", split into shards that hold all samples associated
  with mutually-exclusive target classes.
* "biased", split the dataset through random sampling according
  to a shard-specific random labels distribution.

Utilities provided here are limited to:

* (>=2-)d dataset that be directly loaded into numpy arrays or sparse matrices.
* Single-label, multinomial classification problems.
"""

import os
from typing import Optional, Tuple, Union

import fire  # type: ignore
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix  # type: ignore

from declearn.dataset.examples import load_mnist
from declearn.dataset.utils import (
    load_data_array,
    save_data_array,
    split_multi_classif_dataset,
)

__all__ = [
    "split_data",
]


def load_data(
    data: Optional[str] = None,
    target: Optional[Union[str, int]] = None,
) -> Tuple[Union[np.ndarray, spmatrix], np.ndarray]:
    """Load a dataset in memory from provided path(s).

    This functions supports `.csv`, `.npy`, `.svmlight` and `.sparse`
    file formats. See [declearn.dataset.utils.load_data_array][] for
    details.

    Arguments
    ---------
    data: str or None, default=None
        Path to the data file to import.
        If None, default to importing the MNIST train dataset.
    target: str or int or None, default=None
        If str, path to the labels file to import, or name of a `data`
        column to use as labels (only if `data` points to a csv file).
        If int, index of a `data` column of to use as labels).
        Required if data is not None, ignored if data is None.

    Returns
    -------
    inputs:
        Input features, as a numpy array or scipy sparse matrix.
    labels:
        Ground-truth labels, as a numpy array.
    """
    # Case when no arguments are provided: return the default MNIST dataset.
    if not data:
        return load_mnist(train=True)
    # Otherwise, load the dataset, then load or extract the target labels.
    inputs = load_data_array(data)
    if isinstance(target, str):
        # Case when 'target' points to a separate data file.
        if os.path.isfile(target):
            labels = load_data_array(target)
            if isinstance(labels, spmatrix):
                labels = labels.toarray()  # type: ignore
            elif isinstance(labels, pd.DataFrame):
                labels = labels.values
        # Case when 'target' is the name of a column in a csv file.
        elif isinstance(inputs, pd.DataFrame) and target in inputs:
            labels = inputs.pop(target).values  # type: ignore
            inputs = inputs.values
        else:
            raise ValueError(
                "Invalid 'target' value: either the file is missing, or it "
                "points to a column that is not present in the loaded data."
            )
    elif isinstance(target, int):
        # Case when 'target' is the index of a data column.
        inputs, labels = _extract_column_by_index(
            inputs,  # type: ignore
            target,
        )
    else:
        raise TypeError("Invalid type for 'target': should be str or int.")
    return inputs, labels  # type: ignore


def _extract_column_by_index(
    inputs: Union[np.ndarray, spmatrix, pd.DataFrame],
    target: int,
) -> Tuple[Union[np.ndarray, spmatrix], np.ndarray]:
    """Backend to extract a column by index in a data array."""
    if target > inputs.shape[1]:
        raise ValueError(
            f"Invalid 'target' value: index {target} is out of range "
            f"for the dataset, that has {inputs.shape[1]} columns."
        )
    if isinstance(inputs, pd.DataFrame):
        inputs = inputs.values
    if isinstance(inputs, np.ndarray):
        labels = inputs[:, target]
        inputs = np.delete(inputs, target, axis=1)
    elif isinstance(inputs, spmatrix):
        labels = inputs.getcol(target).toarray().ravel()
        csc = inputs.tocsc()  # type: ignore
        # csc: sparse matrix with efficient column slicing
        idx = [i for i in range(inputs.shape[1]) if i != target]
        inputs = type(inputs)(csc[:, idx])  # type: ignore
    else:  # pragma: no cover
        raise TypeError("Invalid type for 'inputs'.")
    return inputs, labels  # type: ignore


# pylint: disable-next=too-many-positional-arguments
def split_data(  # noqa: PLR0913
    folder: str = ".",
    data_file: Optional[str] = None,
    label_file: Optional[Union[str, int]] = None,
    n_shards: int = 3,
    scheme: str = "iid",
    perc_train: float = 0.8,
    seed: Optional[int] = None,
) -> None:
    """Randomly split a dataset into shards.

    The resulting folder structure is:

        folder/
        └─── data*/
            └─── client*/
            │      train_data.* - training data
            │      train_target.* - training labels
            │      valid_data.* - validation data
            │      valid_target.* - validation labels
            └─── client*/
            │    ...

    Parameters
    ----------
    folder: str, default = "."
        Path to the folder where to add a data folder
        holding output shard-wise files
    data_file: str or None, default=None
        Optional path to a folder where to find the data.
        If None, default to the MNIST example.
    label_file: str or int or None, default=None
        If str, path to the labels file to import, or name of a `data`
        column to use as labels (only if `data` points to a csv file).
        If int, index of a `data` column of to use as labels).
        Required if data is not None, ignored if data is None.
    n_shards: int
        Number of shards between which to split the data.
    scheme: {"iid", "labels", "biased"}, default="iid"
        Splitting scheme(s) to use. In all cases, shards contain mutually-
        exclusive samples and cover the full raw training data.
        - If "iid", split the dataset through iid random sampling.
        - If "labels", split into shards that hold all samples associated
        with mutually-exclusive target classes.
        - If "biased", split the dataset through random sampling according
        to a shard-specific random labels distribution.
    perc_train: float, default= 0.8
        Train/validation split in each client dataset, must be in the
        ]0,1] range.
    seed: int or None, default=None
        Optional seed to the RNG used for all sampling operations.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    # Select output folder.
    folder = os.path.join(folder, f"data_{scheme}")
    # Value-check the 'perc_train' parameter.
    if not (isinstance(perc_train, float) and (0.0 < perc_train <= 1.0)):
        raise ValueError("'perc_train' should be a float in ]0,1]")
    # Load the dataset and split it.
    inputs, labels = load_data(data_file, label_file)
    print(
        f"Splitting data into {n_shards} shards using the '{scheme}' scheme."
    )
    split = split_multi_classif_dataset(
        dataset=(inputs, labels),
        n_shards=n_shards,
        scheme=scheme,  # type: ignore
        p_valid=(1 - perc_train),
        seed=seed,
    )
    # Export the resulting shard-wise data to files.
    for idx, ((x_train, y_train), (x_valid, y_valid)) in enumerate(split):
        subdir = os.path.join(folder, f"client_{idx}")
        os.makedirs(subdir, exist_ok=True)
        save_data_array(os.path.join(subdir, "train_data"), x_train)
        save_data_array(os.path.join(subdir, "train_target"), y_train)
        if x_valid.shape[0]:
            save_data_array(os.path.join(subdir, "valid_data"), x_valid)
            save_data_array(os.path.join(subdir, "valid_target"), y_valid)


def main() -> None:
    "Fire-wrapped `split_data`."
    fire.Fire(split_data)


if __name__ == "__main__":
    main()
