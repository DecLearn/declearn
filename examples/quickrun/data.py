# coding: utf-8

"""Script to download and split MNIST data into heterogeneous shards."""

import argparse
import io
import json
import os
import re
import sys
import tempfile
import textwrap
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import requests  # type: ignore

SOURCE_URL = "https://pjreddie.com/media/files/"

# TODO reduce arg numbers in functions using SplitConfig


def load_mnist(
    folder: Optional[str] = None,
    train: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the raw MNIST dataset, downloading it if needed.

    Arguments
    ---------
    folder: str or None, default=None
        Optional path to a root folder where to find or download the
        raw MNIST data. If None, use a temporary folder.
    train: bool, default=True
        Whether to return the 60k training subset, or the 10k testing one.
    """
    # Optionally use a temporary folder where to download the raw data.
    if folder is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            return load_mnist(tmpdir, train)
    # Load the desired subset of MNIST
    tag = "train" if train else "test"
    url = f"{SOURCE_URL}mnist_{tag}.csv"
    data = requests.get(url, verify=False, timeout=20).content
    df = pd.read_csv(io.StringIO(data.decode("utf-8")), header=None, sep=",")
    return df.iloc[:, 1:].to_numpy(), df[0].to_numpy()[:, None]


def split_mnist(
    folder: str,
    n_shards: int,
    scheme: Literal["iid", "labels", "biased"],
    seed: Optional[int] = None,
    mnist: Optional[str] = None,
    use_csv: bool = False,
) -> None:
    """Download and randomly split the MNIST dataset into shards.

    Parameters
    ----------
    folder: str
        Path to the folder where to export shard-wise files.
    n_shards: int
        Number of shards between which to split the MNIST training data.
    scheme: {"iid", "labels", "biased"}
        Splitting scheme to use. In all cases, shards contain mutually-
        exclusive samples and cover the full raw training data.
        - If "iid", split the dataset through iid random sampling.
        - If "labels", split into shards that hold all samples associated
        with mutually-exclusive target classes.
        - If "biased", split the dataset through random sampling according
        to a shard-specific random labels distribution.
    seed: int or None, default=None
        Optional seed to the RNG used for all sampling operations.
    mnist: str or None, default=None
        Optional path to a folder where to find or download the raw MNIST
        data. If None, use a temporary folder.
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
    inputs, target = load_mnist(mnist, train=True)
    os.makedirs(folder, exist_ok=True)
    print(f"Splitting MNIST into {n_shards} shards using the {scheme} scheme")
    split = func(inputs, target, n_shards, rng)
    # Export the resulting shard-wise data to files.
    for idx, (inp, tgt) in enumerate(split):
        if use_csv:
            path = os.path.join(folder, f"shard_{idx}.csv")
            export_shard_to_csv(path, inp, tgt)
        else:
            np.save(os.path.join(folder, f"shard_{idx}_inputs.npy"), inp)
            np.save(os.path.join(folder, f"shard_{idx}_target.npy"), tgt)


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


def export_shard_to_csv(
    path: str,
    inputs: np.ndarray,
    target: np.ndarray,
) -> None:
    """Export an MNIST shard to a csv file."""
    specs = {"dtype": inputs.dtype.char, "shape": list(inputs[0].shape)}
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"{json.dumps(specs)},target")
        for inp, tgt in zip(inputs, target):
            file.write(f"\n{inp.tobytes().hex()},{int(tgt)}")


def load_mnist_from_csv(
    path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reload an MNIST shard from a csv file."""
    # Prepare data containers.
    inputs = []  # type: List[np.ndarray]
    target = []  # type: List[int]
    # Parse the csv file.
    with open(path, "r", encoding="utf-8") as file:
        # Parse input features' specs from the csv header.
        specs = json.loads(file.readline().rsplit(",", 1)[0])
        dtype = specs["dtype"]
        shape = specs["shape"]
        # Iteratively deserialize features and labels from rows.
        for row in file:
            inp, tgt = row.strip("\n").rsplit(",", 1)
            dat = bytes.fromhex(inp)
            inputs.append(np.frombuffer(dat, dtype=dtype).reshape(shape))
            target.append(int(tgt))
    # Assemble the data into numpy arrays and return.
    return np.array(inputs), np.array(target)


def report_download_progress(
    chunk_number: int, chunk_size: int, file_size: int
):
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write("\r0% |{:<64}| {}%".format(bar, int(percent * 100)))


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
    parser.add_argument(
        "--csv",
        default=False,
        dest="use_csv",
        type=bool,
        help="Export data as csv files (for use with Fed-BioMed).",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> None:
    """Run splitting schemes based on commandline-input arguments."""
    cmdargs = parse_args(args)
    for scheme in cmdargs.schemes or ["iid", "labels", "biased"]:
        split_mnist(
            folder=os.path.join(cmdargs.folder, f"mnist_{scheme}"),
            n_shards=cmdargs.n_shards,
            scheme=scheme,
            seed=cmdargs.seed,
            mnist=cmdargs.folder,
            use_csv=cmdargs.use_csv,
        )


if __name__ == "__main__":
    main()
