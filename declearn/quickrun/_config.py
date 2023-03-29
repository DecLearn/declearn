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

"""TOML-parsable container for quickrun configurations."""

import dataclasses
from typing import Dict, List, Literal, Optional, Union

from declearn.utils import TomlConfig

__all__ = [
    "ModelConfig",
    "DataSplitConfig",
    "ExperimentConfig",
]


@dataclasses.dataclass
class ModelConfig(TomlConfig):
    """Dataclass used to provide custom model location and
    class name"""

    model_file: Optional[str] = None
    model_name: str = "MyModel"


@dataclasses.dataclass
class DataSplitConfig(TomlConfig):
    """Dataclass associated with the function
    declearn.quickrun._split_data:split_data

    export_folder: str
        Path to the folder where to export shard-wise files.
    n_shards: int
        Number of shards between which to split the data.
    data_file: str or None, default=None
        Optional path to a folder where to find the data.
        If None, default to the MNIST example.
    target_file: str or int or None, default=None
        If str, path to the labels file to import. If int, column of
        the data file to be used as labels. Required if data is not None,
        ignored if data is None.
    scheme: {"iid", "labels", "biased"}, default="iid"
        Splitting scheme(s) to use. In all cases, shards contain mutually-
        exclusive samples and cover the full raw training data.
        - If "iid", split the dataset through iid random sampling.
        - If "labels", split into shards that hold all samples associated
        with mutually-exclusive target classes.
        - If "biased", split the dataset through random sampling according
        to a shard-specific random labels distribution.
    perc_train:  float, default= 0.8
        Train/validation split in each client dataset, must be in the
        ]0,1] range.
    seed: int or None, default=None
        Optional seed to the RNG used for all sampling operations.
    """

    export_folder: str = "."
    n_shards: int = 5
    data_file: Optional[str] = None
    label_file: Optional[Union[str, int]] = None
    scheme: Literal["iid", "labels", "biased"] = "iid"
    perc_train: float = 0.8
    seed: Optional[int] = None


@dataclasses.dataclass
class ExperimentConfig(TomlConfig):
    """

    Dataclass associated with the function
    declearn.quickrun._parser:parse_data_folder

    data_folder : str or none
        Absolute path to the main folder hosting the data, overwriting
        the folder argument if provided. If None, default to expected
        prefix search in folder.
    client_names: list or None
        List of custom client names to look for in the data_folder.
        If None, default to expected prefix search.
    dataset_names: dict or None
        Dict of custom dataset names, to look for in each client folder.
        Expect 'train_data, train_target, valid_data, valid_target' as keys.
        If None, , default to expected prefix search.
    """

    data_folder: Optional[str] = None
    client_names: Optional[List[str]] = None
    dataset_names: Optional[Dict[str, str]] = None
