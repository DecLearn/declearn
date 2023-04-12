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

"""Script to split data into heterogeneous shards and save them.

Available splitting scheme:

* "iid", split the dataset through iid random sampling.
* "labels", split into shards that hold all samples associated
with mutually-exclusive target classes.
* "biased", split the dataset through random sampling according
to a shard-specific random labels distribution.

Utilities provided here are limited to:

* 2D Dataset that be directly loaded into numpy arrays, excluding for
instance sparse data
* Single-class classification problems
"""

import os
from typing import Optional, Tuple, Union

import fire  # type: ignore
import numpy as np

from declearn.dataset.examples import load_mnist
from declearn.dataset.utils import load_data_array, split_multi_classif_dataset


def load_data(
    data: Optional[str] = None,
    target: Optional[Union[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a dataset in memory from provided path(s). Requires
    inputs type that can be recognised as array by numpy.

    Arguments
    ---------
    data: str or None, default=None
        Path to the data file to import.
        If None, default to importing the MNIST train dataset.
    target: str or int or None, default=None
        If str, path to the labels file to import. If int, column of
        the data file to be used as labels. Required if data is not None,
        ignored if data is None.

    Note
    -----
    Sparse inputs will not be properly parsed by numpy.
    Currently, this function only works with .csv and .npy files

    """
    if not data:
        return load_mnist(train=True)

    if os.path.isfile(data):
        inputs = load_data_array(data)
        inputs = np.asarray(inputs)
    else:
        print("\n\n", data, "\n\n")
        raise ValueError("The data path provided is not a valid file")

    if isinstance(target, int):
        labels = inputs[:, target]
        inputs = np.delete(inputs, target, axis=1)
    if isinstance(target, str):
        if os.path.isfile(target):
            labels = load_data_array(target)
            labels = np.asarray(labels)
    else:
        raise ValueError(
            "The target provided is invalid, please provide a"
            "valid path to a file with labels or indicate"
            "which column to use as label in the inputs "
        )
    return inputs, labels


# pylint: disable=too-many-arguments,too-many-locals
def split_data(
    folder: str = ".",
    data_file: Optional[str] = None,
    label_file: Optional[Union[str, int]] = None,
    n_shards: int = 3,
    scheme: str = "iid",
    perc_train: float = 0.8,
    seed: Optional[int] = None,
) -> None:
    """Randomly split a dataset into shards.

    The resulting folder structure is :
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
    target_file: str or int or None, default=None
        If str, path to the labels file to import. If int, column of
        the data file to be used as labels. Required if data is not None,
        ignored if data is None.
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
    perc_train:  float, default= 0.8
        Train/validation split in each client dataset, must be in the
        ]0,1] range.
    seed: int or None, default=None
        Optional seed to the RNG used for all sampling operations.
    """
    # Select output folder.
    folder = os.path.join(folder, f"data_{scheme}")
    # Value-check the 'perc_train' parameter.
    if not 0.0 < perc_train <= 1.0:
        raise ValueError("'perc_train' should be a float in ]0,1]")
    # Load the dataset and split it.
    inputs, labels = load_data(data_file, label_file)
    print(
        f"Splitting data into {n_shards} shards "
        f"using the '{scheme}' scheme."
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
        np.save(os.path.join(subdir, "train_data.npy"), x_train)
        np.save(os.path.join(subdir, "train_target.npy"), y_train)
        if len(x_valid):
            np.save(os.path.join(subdir, "valid_data.npy"), x_valid)
            np.save(os.path.join(subdir, "valid_target.npy"), y_valid)


def main():
    "fire-wrapped split data"
    fire.Fire(split_data)


if __name__ == "__main__":
    main()
