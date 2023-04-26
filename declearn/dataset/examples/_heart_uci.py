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

"""Util to download and pre-process the UCI Heart Disease dataset."""

import os
from typing import Literal, Optional, Tuple

import pandas as pd  # type: ignore

__all__ = [
    "load_heart_uci",
]


def load_heart_uci(
    name: Literal["cleveland", "hungarian", "switzerland", "va"],
    folder: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load and/or download a pre-processed UCI Heart Disease dataset.

    See [https://archive.ics.uci.edu/ml/datasets/Heart+Disease] for
    information on the UCI Heart Disease dataset.

    Arguments
    ---------
    name: str
        Name of a center, the dataset from which to return.
    folder: str or None, default=None
        Optional path to a folder where to write output csv files.
        If the file already exists in that folder, read from it.

    Returns
    -------
    data: pd.DataFrame
        Pre-processed dataset from the `name` center.
        May be passed as `data` of a declearn `InMemoryDataset`.
    target: str
        Name of the target column in `data`.
        May be passed as `target` of a declearn `InMemoryDataset`.
    """
    # If the file already exists, read and return it.
    if folder is not None:
        path = os.path.join(folder, f"data_{name}.csv")
        if os.path.isfile(path):
            data = pd.read_csv(path)
            return data, "num"
    # Otherwise, download and pre-process the data, and optionally save it.
    data = download_heart_uci_shard(name)
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        data.to_csv(path, index=False)
    return data, "num"


def download_heart_uci_shard(
    name: Literal["cleveland", "hungarian", "switzerland", "va"],
) -> pd.DataFrame:
    """Download and preprocess a subset of the Heart UCI dataset."""
    print(f"Downloading Heart Disease UCI dataset from center {name}.")
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        f"heart-disease/processed.{name}.data"
    )
    # Download the dataaset.
    data = pd.read_csv(url, header=None, na_values="?")
    columns = [
        # fmt: off
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    data = data.set_axis(columns, axis=1, copy=False)
    # Drop unused columns and rows with missing values.
    data.drop(columns=["ca", "chol", "fbs", "slope", "thal"], inplace=True)
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    # Normalize quantitative variables.
    for col in ("age", "trestbps", "thalach", "oldpeak"):
        data[col] = (  # type: ignore
            data[col] - data[col].mean() / data[col].std()  # type: ignore
        )
    # Binarize the target variable.
    data["num"] = (data["num"] > 0).astype(int)
    # Return the prepared dataframe.
    return data
