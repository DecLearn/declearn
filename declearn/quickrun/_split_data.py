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
from typing import List, Optional, Tuple, Union

import numpy as np

from declearn.dataset.examples import load_mnist
from declearn.dataset.utils import load_data_array
from declearn.quickrun._config import DataSplitConfig


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


def _split_iid(
    inputs: np.ndarray,
    target: np.ndarray,
    n_shards: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a dataset into shards using iid sampling."""
    order = rng.permutation(len(inputs))
    s_len = len(inputs) // n_shards
    split = []  # type: List[Tuple[np.ndarray, np.ndarray]]
    for idx in range(n_shards):
        srt = idx * s_len
        end = (srt + s_len) if idx < (n_shards - 1) else len(order)
        shard = order[srt:end]
        split.append((inputs[shard], target[shard]))
    return split


def _split_labels(
    inputs: np.ndarray,
    target: np.ndarray,
    n_shards: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a dataset into shards with mutually-exclusive label classes."""
    classes = np.unique(target)
    if n_shards > len(classes):
        raise ValueError(
            f"Cannot share {len(classes)} classes between {n_shards}"
            "shards with mutually-exclusive labels."
        )
    s_len = len(classes) // n_shards
    order = rng.permutation(classes)
    split = []  # type: List[Tuple[np.ndarray, np.ndarray]]
    for idx in range(n_shards):
        srt = idx * s_len
        end = (srt + s_len) if idx < (n_shards - 1) else len(order)
        shard = np.isin(target, order[srt:end])
        shuffle = rng.permutation(shard.sum())
        split.append((inputs[shard][shuffle], target[shard][shuffle]))
    return split


def _split_biased(
    inputs: np.ndarray,
    target: np.ndarray,
    n_shards: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a dataset into shards with heterogeneous label distributions."""
    classes = np.unique(target)
    index = np.arange(len(target))
    s_len = len(target) // n_shards
    split = []  # type: List[Tuple[np.ndarray, np.ndarray]]
    for idx in range(n_shards):
        if idx < (n_shards - 1):
            # Draw a random distribution of labels for this node.
            logits = np.exp(rng.normal(size=len(classes)))
            lprobs = logits[target[index]]
            lprobs = lprobs / lprobs.sum()
            # Draw samples based on this distribution, without replacement.
            shard = rng.choice(index, size=s_len, replace=False, p=lprobs)
            index = index[~np.isin(index, shard)]
        else:
            # For the last node: use the remaining samples.
            shard = index
        split.append((inputs[shard], target[shard]))
    return split


def split_data(data_config: DataSplitConfig, folder: str) -> None:
    """Download and randomly split a dataset into shards.

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
    data_config: DataSplitConfig
        A DataSplitConfig instance, see class documentation for details
    """

    def np_save(folder, data, i, name):
        data_dir = os.path.join(folder, f"client_{i}")
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, f"{name}.npy"), data)

    # Overwrite default folder if provided
    scheme = data_config.scheme
    name = f"data_{scheme}"
    data_file = data_config.data_file
    label_file = data_config.label_file
    if data_config.data_folder:
        folder = os.path.dirname(data_config.data_folder)
        name = os.path.split(data_config.data_folder)[-1]
        data_file = os.path.abspath(data_config.data_file)
        label_file = os.path.abspath(data_config.label_file)
    # Select the splitting function to be used.
    if scheme == "iid":
        func = _split_iid
    elif scheme == "labels":
        func = _split_labels
    elif scheme == "biased":
        func = _split_biased
    else:
        raise ValueError(f"Invalid 'scheme' value: '{scheme}'.")
    # Set up the RNG, download the raw dataset and split it.
    rng = np.random.default_rng(data_config.seed)

    inputs, labels = load_data(data_file, label_file)
    print(
        f"Splitting data into {data_config.n_shards}"
        f"shards using the {scheme} scheme"
    )
    split = func(inputs, labels, data_config.n_shards, rng)
    # Export the resulting shard-wise data to files.
    folder = os.path.join(folder, name)
    for i, (inp, tgt) in enumerate(split):
        perc_train = data_config.perc_train
        if not perc_train:
            np_save(folder, inp, i, "train_data")
            np_save(folder, tgt, i, "train_target")
        else:
            if perc_train > 1.0 or perc_train < 0.0:
                raise ValueError("perc_train should be a float in ]0,1]")
            n_train = round(len(inp) * perc_train)
            t_inp, t_tgt = inp[:n_train], tgt[:n_train]
            v_inp, v_tgt = inp[n_train:], tgt[n_train:]
            np_save(folder, t_inp, i, "train_data")
            np_save(folder, t_tgt, i, "train_target")
            np_save(folder, v_inp, i, "valid_data")
            np_save(folder, v_tgt, i, "valid_target")
