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

"""Data-preparation script for the MNIST dataset."""

import os
from typing import Literal, Optional

import fire  # type: ignore

from declearn.dataset.examples import load_mnist
from declearn.dataset.utils import (
    save_data_array,
    split_multi_classif_dataset,
)


DATADIR = os.path.join(os.path.dirname(__file__), "data")


def prepare_mnist(
    nb_clients: int,
    scheme: Literal["iid", "labels", "biased"] = "iid",
    folder: str = DATADIR,
    seed: Optional[int] = None,
) -> str:
    """Fetch and split the MNIST dataset to use it federatively.

    Parameters
    ----------
    nb_clients:
        Number of shards between which to split the raw MNIST data.
    scheme:
        Splitting scheme to use. In all cases, shards contain mutually-
        exclusive samples and cover the full dataset. See details below.
    folder:
        Path to the root folder where to export the raw and split data,
        using adequately-named subfolders.
    seed:
        Optional seed to the RNG used for all sampling operations.

    Data-splitting schemes
    ----------------------

    - If "iid", split the dataset through iid random sampling.
    - If "labels", split into shards that hold all samples associated
      with mutually-exclusive target classes.
    - If "biased", split the dataset through random sampling according
      to a shard-specific random labels distribution.
    """
    # Download (or reload) the raw MNIST data.
    datadir_raw = os.path.join(folder, "mnist_raw")
    dataset_raw = load_mnist(train=True, folder=datadir_raw)
    # Split it based on the input arguments.
    print(f"Splitting MNIST into {nb_clients} shards using '{scheme}' scheme.")
    split_data = split_multi_classif_dataset(
        dataset_raw, n_shards=nb_clients, scheme=scheme, seed=seed
    )
    # Export shard data into expected folder structure.
    folder = os.path.join(folder, f"mnist_{scheme}")
    for idx, ((x_t, y_t), (x_v, y_v)) in enumerate(split_data):
        save_data_array(
            os.path.join(folder, f"client_{idx}", "train_data"), x_t
        )
        save_data_array(
            os.path.join(folder, f"client_{idx}", "train_target"), y_t
        )
        save_data_array(
            os.path.join(folder, f"client_{idx}", "valid_data"), x_v
        )
        save_data_array(
            os.path.join(folder, f"client_{idx}", "valid_target"), y_v
        )
    # Return the path to the split data folder.
    return folder


if __name__ == "__main__":
    fire.Fire(prepare_mnist)
