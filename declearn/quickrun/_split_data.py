# coding: utf-8

"""Script to split data into heterogeneous shards and save them.

Available splitting scheme:

* "iid", split the dataset through iid random sampling.
* "labels", split into shards that hold all samples associated
with mutually-exclusive target classes.
* "biased", split the dataset through random sampling according
to a shard-specific random labels distribution.

Utilities provided here are limited to :

* 2D Dataset that be directly loaded into numpy arrays, excluding for
instance sparse data
* Single-class classification problems

"""

import argparse
import io
import json
import os
import re
import textwrap
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests  # type: ignore

from declearn.dataset import load_data_array

SOURCE_URL = "https://pjreddie.com/media/files"
DEFAULT_FOLDER = "./examples/quickrun/data"
# TODO remove duplicate with _run.py


def load_mnist(
    train: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the raw MNIST dataset.

    Arguments
    ---------
    train: bool, default=True
        Whether to return the 60k training subset, or the 10k testing one.
        Note that the test set should not be used as a validation set.
    """
    # Load the desired subset of MNIST
    tag = "train" if train else "test"
    url = f"{SOURCE_URL}/mnist_{tag}.csv"
    data = requests.get(url, verify=False, timeout=20).content
    df = pd.read_csv(io.StringIO(data.decode("utf-8")), header=None, sep=",")
    return df.iloc[:, 1:].to_numpy(), df[0].to_numpy()[:, None]


def load_data(
    data: Optional[str] = None,
    target: Optional[Union[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a dataset in memory from provided path(s). Requires
    inputs type that can be recognised as array by numpy.

    Arguments
    ---------
    data: str or None, default=None
        Path to the data file to import. If None, default to importing
        the MNIST train dataset.
    target: str or int or None, default=None
        If str, path to the labels file to import. If int, column of
        the data file to be used as labels. Required if data is not None,
        ignored if data is None.

    Note
    -----
    Sparse inputs will not be properly parsed by numpy. Currently, this function
    only works with .csv and .npy files

    """
    if not data:
        return load_mnist()

    if os.path.isfile(data):
        inputs = load_data_array(data)
        inputs = np.asarray(inputs)
    else:
        raise ValueError("The data path provided is not a valid file")

    if isinstance(target, int):
        labels = inputs[target][:, None]
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
        split.append((inputs[shard], target[shard]))
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


def split_data(
    folder: str = DEFAULT_FOLDER,
    n_shards: int = 5,
    data: Optional[str] = None,
    target: Optional[Union[str, int]] = None,
    scheme: Literal["iid", "labels", "biased"] = "iid",
    perc_train: float = 0.8,
    seed: Optional[int] = None,
) -> None:
    """Download and randomly split the MNIST dataset into shards.
    #TODO
    Parameters
    ----------
    folder: str
        Path to the folder where to export shard-wise files.
    n_shards: int
        Number of shards between which to split the MNIST training data.
    data: str or None, default=None
        Optional path to a folder where to find or download the raw MNIST
        data. If None, use a temporary folder.
    scheme: {"iid", "labels", "biased"}, default="iid"
        Splitting scheme to use. In all cases, shards contain mutually-
        exclusive samples and cover the full raw training data.
        - If "iid", split the dataset through iid random sampling.
        - If "labels", split into shards that hold all samples associated
        with mutually-exclusive target classes.
        - If "biased", split the dataset through random sampling according
        to a shard-specific random labels distribution.
    seed: int or None, default=None
        Optional seed to the RNG used for all sampling operations.
    use_csv: bool, default=False
        Whether to export shard-wise csv files rather than pairs of .npy
        files. This uses twice as much disk space and requires using the
        `load_mnist_from_csv` function to reload instead of `numpy.load`
        but is mandatory to have compatibility with the Fed-BioMed API.
    """
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
    rng = np.random.default_rng(seed)
    inputs, labels = load_data(data, target)
    os.makedirs(folder, exist_ok=True)
    print(f"Splitting data into {n_shards} shards using the {scheme} scheme")
    split = func(inputs, labels, n_shards, rng)
    # Export the resulting shard-wise data to files.

    def np_save(data, i, name):
        np.save(os.path.join(folder, f"client_{i}/{name}.npy"), data)

    for i, (inp, tgt) in enumerate(split):
        if not perc_train:
            np_save(inp, i, "train_data")
            np_save(tgt, i, "train_target")
        else:
            if ~(perc_train <= 1.0) or ~(perc_train > 0.0):
                raise ValueError("perc_train should be a float in ]0,1]")
            n_train = round(len(inp) * perc_train)
            t_inp, t_tgt = inp[:n_train], tgt[:n_train]
            v_inp, v_tgt = inp[n_train:], inp[n_train:]
            np_save(t_inp, i, "train_data")
            np_save(t_tgt, i, "train_target")
            np_save(v_inp, i, "valid_data")
            np_save(v_tgt, i, "valid_target")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Set up and run a command-line arguments parser."""
    usage = """
        Download and split MNIST data into heterogeneous shards.

        This script automates the random splitting of the MNIST digits-
        recognition images dataset's 60k training samples into shards,
        based on various schemes. Shards contain mutually-exclusive
        samples and cover the full raw dataset.

        The implemented schemes are the following:
        * "iid":
            Split the dataset through iid random sampling.
        * "labels":
            Split the dataset into shards that hold all samples
            that have mutually-exclusive target classes.
        * "biased":
            Split the dataset through random sampling according
            to a shard-specific random labels distribution.
    """
    usage = re.sub("\n *(?=[a-z])", " ", textwrap.dedent(usage))
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        usage=re.sub("- ", "-", usage),
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=5,
        help="Number of shards between which to split the MNIST training data.",
    )
    parser.add_argument(
        "--root",
        default=".",
        dest="folder",
        help="Path to the root folder where to export raw and split data.",
    )
    parser.add_argument(
        "--data_path",
        default=None,  # CHECK
        dest="data",
        help="Path to the data to be split",
    )
    parser.add_argument(
        "--target_path",
        default=None,  # CHECK
        dest="target",
        help="Path to the labels to be split",
    )
    schemes_help = """
        Splitting scheme(s) to use, among {"iid", "labels", "biased"}.
        If this argument is not specified, all three values are used.
        See details above on the schemes' definition.
    """
    parser.add_argument(
        "--scheme",
        action="append",
        choices=["iid", "labels", "biased"],
        default=["iid"],
        dest="schemes",
        nargs="+",
        help=textwrap.dedent(schemes_help),
    )
    parser.add_argument(
        "--seed",
        default=20221109,
        dest="seed",
        type=int,
        help="RNG seed to use (default: 20221109).",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> None:
    """Run splitting schemes based on commandline-input arguments."""
    cmdargs = parse_args(args)
    for scheme in cmdargs.schemes:
        split_data(
            folder=os.path.join(cmdargs.folder, f"data_{scheme}"),
            n_shards=cmdargs.n_shards,
            data=cmdargs.data,
            target=cmdargs.target,
            scheme=scheme,
            seed=cmdargs.seed,
        )


if __name__ == "__main__":
    main()
