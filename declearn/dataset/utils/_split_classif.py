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

"""Utils to split a multi-category classification dataset into shards."""

from typing import List, Literal, Optional, Tuple

import numpy as np


__all__ = [
    "split_multi_classif_dataset",
]


def split_multi_classif_dataset(
    dataset: Tuple[np.ndarray, np.ndarray],
    n_shards: int,
    scheme: Literal["iid", "labels", "biased"],
    p_valid: float = 0.2,
    seed: Optional[int] = None,
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """Split a classification dataset into (opt. heterogeneous) shards.

    The data-splitting schemes are the following:

    - If "iid", split the dataset through iid random sampling.
    - If "labels", split into shards that hold all samples associated
      with mutually-exclusive target classes.
    - If "biased", split the dataset through random sampling according
      to a shard-specific random labels distribution.

    Parameters
    ----------
    dataset: tuple(np.ndarray, np.ndarray)
        Raw dataset, as a pair of numpy arrays that respectively contain
        the input features and (aligned) labels.
    n_shards: int
        Number of shards between which to split the dataset.
    scheme: {"iid", "labels", "biased"}
        Splitting scheme to use. In all cases, shards contain mutually-
        exclusive samples and cover the full dataset. See details above.
    p_valid: float, default=0.2
        Share of each shard to turn into a validation subset.
    seed: int or None, default=None
        Optional seed to the RNG used for all sampling operations.

    Returns
    -------
    shards: list[((np.ndarray, np.ndarray), (np.ndarray, np.ndarray))]
        List of dataset shards, where each element is formatted as a
        tuple of tuples: `((x_train, y_train), (x_valid, y_valid))`.

    Raises
    ------
    ValueError
        If `scheme` has an invalid value.
    """
    # Select the splitting function to be used.
    if scheme == "iid":
        func = split_iid
    elif scheme == "labels":
        func = split_labels
    elif scheme == "biased":
        func = split_biased
    else:
        raise ValueError(f"Invalid 'scheme' value: '{scheme}'.")
    # Set up the RNG and split the dataset into shards.
    rng = np.random.default_rng(seed)
    inputs, target = dataset
    split = func(inputs, target, n_shards, rng)
    # Further split shards into training and validation subsets, and return.
    return [train_valid_split(inp, tgt, p_valid, rng) for inp, tgt in split]


def split_iid(
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


def split_labels(
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
        split.append((inputs[shard], target[shard]))
    return split


def split_biased(
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


def train_valid_split(
    inputs: np.ndarray,
    target: np.ndarray,
    p_valid: float,
    rng: np.random.Generator,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split a dataset between train and validation using iid sampling."""
    order = rng.permutation(len(inputs))
    v_len = np.ceil(len(inputs) * p_valid).astype(int)
    train = inputs[order[v_len:]], target[order[v_len:]]
    valid = inputs[order[:v_len]], target[order[:v_len]]
    return train, valid
