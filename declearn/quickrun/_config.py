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
from typing import Any, Dict, List, Optional, Union

from declearn.metrics import MetricInputType, MetricSet
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
    """Dataclass associated with the functions
    declearn.quickrun._split_data:split_data and
    declearn.quickrun._parser:parse_data_folder

    data_folder: str
        Absolute path to the folder where to export shard-wise files,
        and/or to the main folder hosting the data.
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
    client_names: list or None
        List of custom client names to look for in the data_folder.
        If None, default to expected prefix search.
    dataset_names: dict or None
        Dict of custom dataset names, to look for in each client folder.
        Expect 'train_data, train_target, valid_data, valid_target' as keys.
        If None, , default to expected prefix search.
    """

    # Common args
    data_folder: Optional[str] = None
    # split_data args
    n_shards: int = 5
    data_file: Optional[str] = None
    label_file: Optional[Union[str, int]] = None
    scheme: str = "iid"
    perc_train: float = 0.8
    seed: Optional[int] = None
    # parse_data_folder args
    client_names: Optional[List[str]] = None
    dataset_names: Optional[Dict[str, str]] = None


@dataclasses.dataclass
class ExperimentConfig(TomlConfig):
    """

    Dataclass providing kwargs to
    declearn.main._server.FederatedServer
    and declearn.main._client.FederatedClient

    metrics: list[str] or None
        List of Metric childclass names, defining evaluation metrics
        to compute in addition to the model's loss.
    checkpoint: str or None
        The checkpoint folder path and use default values for other parameters
        to be used so as to save round-wise model
    """

    metrics: Optional[MetricSet] = None
    checkpoint: Optional[str] = None

    def parse_metrics(
        self,
        inputs: Union[MetricSet, Dict[str, Any], List[MetricInputType], None],
    ) -> Optional[MetricSet]:
        """Parser for metrics."""
        if inputs is None or isinstance(inputs, MetricSet):
            return None
        try:
            # Case of a manual listing of metrics (most expected).
            if isinstance(inputs, (tuple, list)):
                return MetricSet.from_specs(inputs)
            # Case of a MetricSet config dict (unexpected but supported).
            if isinstance(inputs, dict):
                return MetricSet.from_config(inputs)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Failed to parse inputs for field 'metrics': {exc}."
            ) from exc
        raise TypeError(
            "Failed to parse inputs for field 'metrics': unproper type."
        )
