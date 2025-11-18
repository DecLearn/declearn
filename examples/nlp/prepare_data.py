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

"""Data-preparation script for a Hugging Face NLP dataset (e.g. IMDb)."""

import os
import random
from typing import Optional

import fire
import pandas as pd
from datasets import load_dataset

DATADIR = os.path.join(os.path.dirname(__file__), "data")


def prepare_hf_dataset(
    nb_clients: int,
    text_field: str = "text",
    label_field: str = "label",
    folder: str = DATADIR,
    seed: Optional[int] = None,
    split_ratio: float = 0.8,
) -> str:
    """Fetch and split a Hugging Face text dataset for federated learning.

    Parameters
    ----------
    nb_clients:
        Number of clients (shards) to create.
    text_field:
        Field name containing the text samples.
    label_field:
        Field name containing the labels.
    folder:
        Output directory root.
    seed:
        Random seed.
    split_ratio:
        Train/validation split ratio inside each client.

    Returns
    -------
    folder:
        Output directory in which dataset shards are exported
    """
    random.seed(seed)
    os.makedirs(folder, exist_ok=True)

    dataset_name = "imdb"
    print(f"Loading Hugging Face dataset: {dataset_name}")
    # Load the dataset and shuffle it
    dataset = load_dataset(dataset_name, split="train").shuffle(seed)

    # Select only a small proportion of the dataset to reduce
    # example execution time
    cut = int(0.05 * len(dataset))
    dataset = dataset.select(range(cut))

    # Build each client dataset shard
    for client_idx in range(nb_clients):
        client_folder = os.path.join(folder, f"client_{client_idx}")
        os.makedirs(client_folder, exist_ok=True)

        shard_dataset = dataset.shard(num_shards=nb_clients, index=client_idx)
        split_dataset = shard_dataset.train_test_split(train_size=split_ratio)

        x_t = list(split_dataset["train"]["text"])
        y_t = list(split_dataset["train"]["label"])
        x_v = list(split_dataset["test"]["text"])
        y_v = list(split_dataset["test"]["label"])

        # Convert to dataframes
        df_train = pd.DataFrame({text_field: x_t, label_field: y_t})
        df_valid = pd.DataFrame({text_field: x_v, label_field: y_v})

        # Save as csv
        df_train.to_csv(
            os.path.join(client_folder, "train_data.csv"), index=False
        )
        df_valid.to_csv(
            os.path.join(client_folder, "valid_data.csv"), index=False
        )

    print(f"Saved split data to: {folder}")
    return folder


if __name__ == "__main__":
    fire.Fire(prepare_hf_dataset)
