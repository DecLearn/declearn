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

"""Util to download and pre-process the UCI Heart Disease dataset."""

import io
import os
import zipfile
from typing import Literal, Optional, Tuple, Union

import pandas as pd
import requests

__all__ = [
    "load_heart_uci",
]


def load_heart_uci(
    name: Literal["cleveland", "hungarian", "switzerland", "va"],
    folder: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load and/or download a pre-processed UCI Heart Disease dataset.

    See [https://archive.ics.uci.edu/dataset/45/heart+disease] for
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
    # If the pre-processed file already exists, read and return it.
    if folder is not None:
        path = os.path.join(folder, f"data_{name}.csv")
        if os.path.isfile(path):
            data = pd.read_csv(path)
            return data, "num"
    # Download (and optionally save) or read from the source zip file.
    source = get_heart_uci_zipfile(folder)
    # Extract the target shard and preprocess it.
    data = extract_heart_uci_shard(name, source)
    data = preprocess_heart_uci_dataframe(data)
    # Optionally save the preprocessed shard to disk.
    if folder is not None:
        path = os.path.join(folder, f"data_{name}.csv")
        data.to_csv(path, sep=",", encoding="utf-8", index=False)
    return data, "num"


def get_heart_uci_zipfile(folder: Optional[str]) -> Union[str, bytes]:
    """Download and opt. save the Heart Dataset zip file, or return its path.

    Return either the path to the zip file, or its contents.
    """
    # Case when the data is to be downloaded and kept only in memory.
    if folder is None:
        return download_heart_uci()
    # Case when the data can be read from a pre-existing file on disk.
    path = os.path.join(folder, "heart+disease.zip")
    if os.path.isfile(path):
        return path
    # Case when the data is to be donwloaded and saved on disk for re-use.
    data = download_heart_uci()
    with open(path, "wb") as file:
        file.write(data)
    return data


def download_heart_uci() -> bytes:
    """Download the Heart Disease UCI dataset source file."""
    print("Downloading Heart Disease UCI dataset.")
    url = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
    reply = requests.get(url, timeout=300)
    try:
        reply.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            "Failed to download Heart Disease UCI source file."
        ) from exc
    return reply.content


def extract_heart_uci_shard(
    name: Literal["cleveland", "hungarian", "switzerland", "va"],
    source: Union[str, bytes],
) -> pd.DataFrame:
    """Read a subset of the Heart UCI data, from in-memory or on-disk data."""
    zdat = source if isinstance(source, str) else io.BytesIO(source)
    with zipfile.ZipFile(zdat) as archive:  # type: ignore
        with archive.open(f"processed.{name}.data") as path:
            data = pd.read_csv(path, header=None, na_values="?")
    return data


def preprocess_heart_uci_dataframe(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Preprocess a subset of the Heart UCI dataset."""
    # fmt: off
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    # fmt: on
    data = data.set_axis(columns, axis=1, copy=False)
    # Drop unused columns and rows with missing values.
    data.drop(columns=["ca", "chol", "fbs", "slope", "thal"], inplace=True)
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    # Normalize quantitative variables.
    for col in ("age", "trestbps", "thalach", "oldpeak"):
        data[col] = (
            data[col] - data[col].mean() / data[col].std()  # type: ignore
        )
    # Binarize the target variable.
    data["num"] = (data["num"] > 0).astype(int)
    # Return the prepared dataframe.
    return data
