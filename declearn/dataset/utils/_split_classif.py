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

"""Utils to split a multi-category classification dataset into shards."""

import functools
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import scipy.stats  # type: ignore
from scipy.sparse import csr_matrix, spmatrix  # type: ignore

__all__ = [
    "split_multi_classif_dataset",
]


def split_multi_classif_dataset(
    dataset: Tuple[Union[np.ndarray, spmatrix], np.ndarray],
    n_shards: int,
    scheme: Literal["iid", "labels", "dirichlet", "biased"],
    p_valid: float = 0.2,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """Split a classification dataset into (opt. heterogeneous) shards.

    The data-splitting schemes are the following:

    - If "iid", split the dataset through iid random sampling.
    - If "labels", split into shards that hold all samples associated
      with mutually-exclusive target classes.
    - If "dirichlet", split the dataset through random sampling using
      label-wise shard-assignment probabilities drawn from a symmetrical
      Dirichlet distribution, parametrized by an `alpha` parameter.
    - If "biased", split the dataset through random sampling according
      to a shard-specific random labels distribution.

    Parameters
    ----------
    dataset: tuple(np.ndarray|spmatrix, np.ndarray)
        Raw dataset, as a pair of numpy arrays that respectively contain
        the input features and (aligned) labels. Input features may also
        be a scipy sparse matrix, that will temporarily be cast to CSR.
    n_shards: int
        Number of shards between which to split the dataset.
    scheme: {"iid", "labels", "dirichlet", "biased"}
        Splitting scheme to use. In all cases, shards contain mutually-
        exclusive samples and cover the full dataset. See details above.
    p_valid: float, default=0.2
        Share of each shard to turn into a validation subset.
    seed: int or None, default=None
        Optional seed to the RNG used for all sampling operations.
    **kwargs:
        Additional hyper-parameters specific to the split scheme.
        Exhaustive list of possible values:
            - `alpha: float = 0.5` for `scheme="dirichlet"`

    Returns
    -------
    shards:
        List of dataset shards, where each element is formatted as a
        tuple of tuples: `((x_train, y_train), (x_valid, y_valid))`.
        Input features will be of same type as `inputs`.

    Raises
    ------
    TypeError
        If `inputs` is not a numpy array or scipy sparse matrix.
    ValueError
        If `scheme` has an invalid value.
    """
    # Select the splitting function to be used.
    if scheme == "iid":
        func = split_iid
    elif scheme == "labels":
        func = split_labels
    elif scheme == "dirichlet":
        func = functools.partial(
            split_dirichlet, alpha=kwargs.get("alpha", 0.5)
        )
    elif scheme == "biased":
        func = split_biased
    else:
        raise ValueError(f"Invalid 'scheme' value: '{scheme}'.")
    # Set up the RNG and unpack the dataset.
    rng = np.random.default_rng(seed)
    inputs, target = dataset
    # Optionally handle sparse matrix inputs.
    sp_type: Optional[Type[spmatrix]] = None
    if isinstance(inputs, spmatrix):
        sp_type = type(inputs)
        inputs = csr_matrix(inputs)
    elif not isinstance(inputs, np.ndarray):
        raise TypeError(
            "'inputs' should be a numpy array or scipy sparse matrix."
        )
    # Split the dataset into shards.
    split = func(inputs, target, n_shards, rng)
    # Further split shards into training and validation subsets.
    shards = [train_valid_split(inp, tgt, p_valid, rng) for inp, tgt in split]
    # Optionally convert back sparse inputs, then return.
    if sp_type is not None:
        shards = [
            ((sp_type(xt), yt), (sp_type(xv), yv))  # type: ignore
            for (xt, yt), (xv, yv) in shards
        ]
    return shards


def split_iid(
    inputs: Union[np.ndarray, csr_matrix],
    target: np.ndarray,
    n_shards: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a dataset into shards using iid sampling."""
    order = rng.permutation(inputs.shape[0])
    s_len = inputs.shape[0] // n_shards
    split: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in range(n_shards):
        srt = idx * s_len
        end = (srt + s_len) if idx < (n_shards - 1) else len(order)
        shard = order[srt:end]
        split.append((inputs[shard], target[shard]))
    return split


def split_labels(
    inputs: Union[np.ndarray, csr_matrix],
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
    split: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in range(n_shards):
        srt = idx * s_len
        end = (srt + s_len) if idx < (n_shards - 1) else len(order)
        shard = np.isin(target, order[srt:end])
        split.append((inputs[shard], target[shard]))
    return split


def split_biased(
    inputs: Union[np.ndarray, csr_matrix],
    target: np.ndarray,
    n_shards: int,
    rng: np.random.Generator,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a dataset into shards with heterogeneous label distributions.

    Use a normal distribution to draw logits of labels distributions for
    each and every node.

    This approach is not based on the litterature. We advise end-users to
    use a Dirichlet split instead, which is probably better-grounded.
    """
    classes = np.unique(target)
    index = np.arange(len(target))
    s_len = len(target) // n_shards
    split: List[Tuple[np.ndarray, np.ndarray]] = []
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


def split_dirichlet(
    inputs: Union[np.ndarray, csr_matrix],
    target: np.ndarray,
    n_shards: int,
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split a dataset into shards with heterogeneous label distributions.

    Use a symmetrical multinomial Dirichlet(alpha) distribution to sample
    the proportion of samples per label in each shard.

    This approach has notably been used by Sturluson et al. (2021).
    FedRAD: Federated Robust Adaptive Distillation. arXiv:2112.01405 [cs.LG]
    """
    classes = np.unique(target)
    # Draw per-label proportion of samples to assign to each shard.
    process = scipy.stats.dirichlet(alpha=[alpha] * n_shards)
    c_probs = process.rvs(size=len(classes), random_state=rng)
    # Randomly assign label-wise samples to shards based on these.
    shard_i: List[List[int]] = [[] for _ in range(n_shards)]
    for lab_i, label in enumerate(classes):
        index = np.where(target == label)[0]
        s_idx = rng.choice(n_shards, size=len(index), p=c_probs[lab_i])
        for i in range(n_shards):
            shard_i[i].extend(index[s_idx == i])
    # Gather the actual sample shards.
    return [
        (inputs[index], target[index])  # type: ignore
        for index in shard_i
    ]


def train_valid_split(
    inputs: np.ndarray,
    target: np.ndarray,
    p_valid: float,
    rng: np.random.Generator,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split a dataset between train and validation using iid sampling."""
    order = rng.permutation(inputs.shape[0])
    v_len = np.ceil(inputs.shape[0] * p_valid).astype(int)
    train = inputs[order[v_len:]], target[order[v_len:]]
    valid = inputs[order[:v_len]], target[order[:v_len]]
    return train, valid
