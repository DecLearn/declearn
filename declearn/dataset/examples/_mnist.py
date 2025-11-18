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

"""Util to download the MNIST digit-classification dataset."""

import gzip
import os
from typing import Optional, Tuple

import numpy as np
import requests

__all__ = [
    "load_mnist",
]


def load_mnist(
    train: bool = True,
    folder: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and/or download the MNIST digit-classification dataset.

    See [https://en.wikipedia.org/wiki/MNIST_database] for information
    on the MNIST dataset.

    Arguments
    ---------
    train: bool, default=True
        Whether to return the 60k training subset, or the 10k testing one.
    folder: str or None, default=None
        Optional path to a root folder where to find or download the
        raw MNIST data. If None, download the data but only store it
        in memory.

    Returns
    -------
    images: np.ndarray
        Input images, as a (n_images, 28, 28) float numpy array.
        May be passed as `data` of a declearn `InMemoryDataset`.
    labels: np.ndarray
        Target labels, as a (n_images) int numpy array.
        May be passed as `target` of a declearn `InMemoryDataset`.
    """
    tag = "train" if train else "t10k"
    images = _load_mnist_data(folder, tag, images=True)
    labels = _load_mnist_data(folder, tag, images=False)
    return images, labels


def _load_mnist_data(
    folder: Optional[str], tag: str, images: bool
) -> np.ndarray:
    """Load (and/or download) and return data from a raw MNIST file."""
    name = f"{tag}-images-idx3" if images else f"{tag}-labels-idx1"
    name = f"{name}-ubyte.gz"
    # Optionally download the gzipped file from the internet.
    if folder is None or not os.path.isfile(os.path.join(folder, name)):
        try:
            data = _download_mnist_file(name, folder, lecun=True)
        except RuntimeError:  # pragma: no cover
            data = _download_mnist_file(name, folder, lecun=False)
        data = gzip.decompress(data)
    # Otherwise, read its contents from a local copy.
    else:
        with gzip.open(os.path.join(folder, name), "rb") as file:
            data = file.read()
    # Read and parse the source data into a numpy array.
    if images:
        shape, off = (
            [int(data[i : i + 4].hex(), 16) for i in range(4, 16, 4)],
            16,
        )
    else:
        shape, off = [int(data[4:8].hex(), 16)], 8
    array = np.frombuffer(bytearray(data[off:]), dtype="uint8").reshape(shape)
    return (array / 255).astype(np.single) if images else array


def _download_mnist_file(
    name: str,
    folder: Optional[str],
    lecun: bool = True,
) -> bytes:
    """Download a MNIST source file and opt. save it in a given folder."""
    # Download the file in memory.
    print(f"Downloading MNIST source file {name}.")
    base_url = (
        "http://yann.lecun.com/exdb"
        if lecun
        else "https://ossci-datasets.s3.amazonaws.com"
    )
    reply = requests.get(f"{base_url}/mnist/{name}", timeout=300)
    try:
        reply.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Failed to download MNIST source file {name}."
        ) from exc
    # Optionally dump the file to disk.
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, name), "wb") as file:
            file.write(reply.content)
    # Return the downloaded data.
    return reply.content
