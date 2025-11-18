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

"""TOML-parsable container for quickrun configurations."""

import dataclasses
from typing import Any, Dict, List, Optional, Union

from declearn.metrics import MetricInputType, MetricSet
from declearn.utils import TomlConfig

__all__ = [
    "DataSourceConfig",
    "ExperimentConfig",
    "ModelConfig",
]


@dataclasses.dataclass
class ModelConfig(TomlConfig):
    """Dataclass used to provide custom model location and name.

    Attributes
    ----------
    model_file: str or None
        Path to the python file under which the model is declared.
        If None, look for "model.py" parallel to the "config.toml" one.
    model_name: str, default="model"
        Name of the variable storing the declearn Model to train,
        declared in the main scope of the model file.
    """

    model_file: Optional[str] = None
    model_name: str = "model"


@dataclasses.dataclass
class DataSourceConfig(TomlConfig):
    """Dataclass associated with the quickrun's `parse_data_folder` function.

    Attributes
    ----------
    data_folder: str
        Absolute path to the to the main folder hosting the data.
    client_names: list or None
        List of custom client names to look for in the data folder.
        If None, default to all subdirectories of the data folder.
    dataset_names: dict or None
        Dict of custom dataset names, to look for in each client folder.
        Expect 'train_data, train_target, valid_data, valid_target' as keys.
        If None, files will be expected to be prefixed using the former keys.
    """

    data_folder: Optional[str] = None
    client_names: Optional[List[str]] = None
    dataset_names: Optional[Dict[str, str]] = None


@dataclasses.dataclass
class ExperimentConfig(TomlConfig):
    """Dataclass providing kwargs to `FederatedServer` and `FederatedClient`.

    Attributes
    ----------
    metrics: MetricSet or None
        Optional MetricSet instance, defining evaluation metrics to compute
        in addition to the model's loss. It may be parsed from a list of
        Metric names or (name, config) tuples), or from a MetricSet config
        dict.
    checkpoint: str or None
        The checkpoint folder path where to save the server's and client-wise
        outputs (round-wise model weights, evaluation metrics, logs, etc.).
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
