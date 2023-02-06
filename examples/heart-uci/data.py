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

"""Script to download and pre-process the UCI Heart Disease Dataset."""

import argparse
import os
from typing import Collection

import pandas as pd

NAMES = ("cleveland", "hungarian", "switzerland", "va")

COLNAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get_data(
    folder: str = DATADIR,
    names: Collection[str] = NAMES,
) -> None:
    """Download and process the UCI heart disease dataset.

    Arguments
    ---------
    folder: str
        Path to the folder where to write output csv files.
    names: list[str]
        Names of centers, the dataset from which to download,
        pre-process and export as csv files.
    """
    for name in names:
        print(f"Downloading data from center {name}:")
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            f"heart-disease/processed.{name}.data"
        )
        print(url)
        # Download the dataset.
        df = pd.read_csv(url, header=None, na_values="?")
        df.columns = COLNAMES
        # Drop unused columns and rows with missing values.
        df.drop(columns=["ca", "chol", "fbs", "slope", "thal"], inplace=True)
        df.dropna(inplace=True)
        # Normalize quantitative variables.
        for col in ("age", "trestbps", "thalach", "oldpeak"):
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        # Binarize the target variable.
        df["num"] = (df["num"] > 0).astype(int)
        # Export the resulting dataset to a csv file.
        os.makedirs(folder, exist_ok=True)
        df.to_csv(f"{folder}/{name}.csv", index=False)


# Code executed when the script is called directly.
if __name__ == "__main__":
    # Parse commandline parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default=DATADIR,
        help="folder where to write output csv files",
    )
    parser.add_argument(
        "names",
        action="append",
        nargs="+",
        help="name(s) of client center(s), data from which to prepare",
        choices=["cleveland", "hungarian", "switzerland", "va"],
    )
    args = parser.parse_args()
    # Download and pre-process the selected dataset(s).
    get_data(folder=args.folder, names=args.names)
