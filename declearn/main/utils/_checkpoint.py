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

"""Model and metrics checkpointing util."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd  # type: ignore
from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.model.api import Model
from declearn.optimizer import Optimizer
from declearn.utils import (
    deserialize_object,
    json_dump,
    json_load,
    serialize_object,
)

__all__ = [
    "Checkpointer",
]


class Checkpointer:
    """Model, optimizer, and metrics checkpointing class.

    This class provides with basic checkpointing capabilities, that
    enable saving a Model, an Optimizer and a dict of metric results
    at various points throughout an experiment, and reloading these
    checkpointed states and results.

    The key method is `checkpoint`, that enables saving all three types
    of objects at once and tagging them with a single timestamp label.
    Note that its `first_call` bool parameter should be set to True on
    the first call, to ensure the model's and optimizer's configurations
    are saved in addition to their states, and preventing the metrics
    from being appended to files from a previous experiment.

    Other methods are exposed that provide with targetted saving and
    loading: `save_model`, `save_optimizer`, `save_metrics` and their
    counterparts `load_model`, `load_optimizer` and `load_metrics`.
    Note that the latter may either be used to load metrics at a given
    timestamp, or their entire history.
    """

    def __init__(
        self,
        folder: str,
        max_history: Optional[int] = None,
    ) -> None:
        """Instantiate the checkpointer.

        Parameters
        ----------
        folder: str
            Folder where to write output save files.
        max_history: int or None, default=None
            Maximum number of model and optimizer state save files to keep.
            Older files are garbage-collected. If None, keep all files.
        """
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)
        if max_history is not None:
            if not (isinstance(max_history, int) and max_history >= 0):
                raise TypeError("'max_history' must be a positive int or None")
        self.max_history = max_history

    @classmethod
    def from_specs(
        cls,
        inputs: Union[str, Dict[str, Any], Self],
    ) -> Self:
        """Type-check and/or transform inputs into a Checkpointer instance.

        This classmethod is merely implemented to avoid duplicate and
        boilerplate code from polluting FL orchestrating classes.

        Parameters
        ----------
        specs: Checkpointer or dict[str, any] or str
            Checkpointer instance to type-check, or instantiation kwargs
            to parse into one. If a single string is passed, treat it as
            the `folder` argument, and use default other parameters.

        Returns
        -------
        checkpointer: Checkpointer
            Checkpointer instance, type-checked or instantiated from inputs.

        Raises
        ------
        TypeError:
            If `inputs` is of unproper type.
        Other exceptions may be raised when calling this class's `__init__`.
        """
        if isinstance(inputs, str):
            inputs = {"folder": inputs}
        if isinstance(inputs, dict):
            inputs = cls(**inputs)
        if not isinstance(inputs, cls):
            raise TypeError("'inputs' should be a Checkpointer, dict or str.")
        return inputs

    # utility methods

    def garbage_collect(
        self,
        prefix: str,
    ) -> None:
        """Delete files with matching prefix based on self.max_history.

        Sort files starting with `prefix` under `self.folder`, and if
        there are more than `self.max_history`, delete the first ones.
        Files are expected to be named as "{prefix}_{timestamp}.{ext}"
        so that doing so will remove the older files.

        Parameters
        ----------
        prefix: str
            Prefix based on which to filter files under `self.folder`.
        """
        if self.folder and self.max_history:
            files = self.sort_matching_files(prefix)
            for idx in range(0, len(files) - self.max_history):
                os.remove(os.path.join(self.folder, files[idx]))

    def sort_matching_files(
        self,
        prefix: str,
    ) -> List[str]:
        """Return the sorted of files under `self.folder` with a given prefix.

        Parameters
        ----------
        prefix: str
            Prefix based on which to filter files under `self.folder`.

        Returns
        -------
        fnames: list[str]
            Sorted list of names of files under `self.folder` that start
            with `prefix`.
        """
        fnames = [f for f in os.listdir(self.folder) if f.startswith(prefix)]
        return sorted(fnames)

    # saving methods

    def save_model(
        self,
        model: Model,
        config: bool = True,
        state: bool = True,
        timestamp: Optional[str] = None,
    ) -> Optional[str]:
        """Save a Model's configuration and/or weights to JSON files.

        Also garbage-collect existing files based on self.max_history.

        Parameters
        ----------
        model: Model
            Model instance to save.
        config: bool, default=True
            Flag indicating whether to save the model's config to a file.
        state: bool, default=True
            Flag indicating whether to save the model's weights to a file.
        timestamp: str or None, default=None
            Optional preset timestamp to add as weights file suffix.

        Returns
        -------
        timestamp: str or None
            Timestamp string labeling the output weights file, if any.
            If `states is None`, return None.
        """
        model_config = (
            None
            if not config
            else (serialize_object(model, allow_unregistered=True).to_dict())
        )
        return self._save_object(
            prefix="model",
            config=model_config,
            states=model.get_weights() if state else None,
            timestamp=timestamp,
        )

    def save_optimizer(
        self,
        optimizer: Optimizer,
        config: bool = True,
        state: bool = True,
        timestamp: Optional[str] = None,
    ) -> Optional[str]:
        """Save an Optimizer's configuration and/or state to JSON files.

        Parameters
        ----------
        optimizer: Optimizer
            Optimizer instance to save.
        config: bool, default=True
            Flag indicating whether to save the optimizer's config to a file.
        state: bool, default=True
            Flag indicating whether to save the optimizer's state to a file.
        timestamp: str or None, default=None
            Optional preset timestamp to add as state file suffix.

        Returns
        -------
        timestamp: str or None
            Timestamp string labeling the output states file, if any.
            If `states is None`, return None.
        """
        return self._save_object(
            prefix="optimizer",
            config=optimizer.get_config() if config else None,
            states=optimizer.get_state() if state else None,
            timestamp=timestamp,
        )

    def _save_object(
        self,
        prefix: str,
        config: Any = None,
        states: Any = None,
        timestamp: Optional[str] = None,
    ) -> Optional[str]:
        """Shared backend for `save_model` and `save_optimizer`.

        Parameters
        ----------
        prefix: str
            Prefix to the created file(s).
            Also used to garbage-collect state files.
        config: object or None, default=None
            Optional JSON-serializable config to save.
            Output file will be named "{prefix}.json".
        states: object or None, default=None
            Optional JSON-serializable data to save.
            Output file will be named "{prefix}_{timestamp}.json".
        timestamp: str or None, default=None
            Optional preset timestamp to add as state file suffix.
            If None, generate a timestamp to use.

        Returns
        -------
        timestamp: str or None
            Timestamp string labeling the output states file, if any.
            If `states is None`, return None.
        """
        if config:
            fpath = os.path.join(self.folder, f"{prefix}_config.json")
            json_dump(config, fpath)
        if states is not None:
            if timestamp is None:
                timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            fpath = os.path.join(
                self.folder, f"{prefix}_state_{timestamp}.json"
            )
            json_dump(states, fpath)
            self.garbage_collect(f"{prefix}_state")
            return timestamp
        return None

    def save_metrics(
        self,
        metrics: Dict[str, Union[float, np.ndarray]],
        prefix: str = "metrics",
        append: bool = True,
        timestamp: Optional[str] = None,
    ) -> str:
        """Save a dict of metrics to a csv and a json files.

        Parameters
        ----------
        metrics: dict[str, (float | np.ndarray)]
            Dict storing metric values that need saving.
            Note that numpy arrays will be converted to lists.
        prefix: str, default="metrics"
            Prefix to the output files' names.
        append: bool, default=True
            Whether to append to the files in case they already exist.
            If False, overwrite any existing file.
        timestamp: str or None, default=None
            Optional preset timestamp to associate with the metrics.

        Returns
        -------
        timestamp: str
            Timestamp string labelling the checkpointed metrics.
        """
        # Set up a timestamp and convert metrics to raw-JSON-compatible values.
        if timestamp is None:
            timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        scores = {
            key: val.tolist() if isinstance(val, np.ndarray) else float(val)
            for key, val in metrics.items()
        }
        # Filter out scalar metrics and write them to a csv file.
        scalars = {k: v for k, v in scores.items() if isinstance(v, float)}
        fpath = os.path.join(self.folder, f"{prefix}.csv")
        pd.DataFrame(scalars, index=[timestamp]).to_csv(
            fpath,
            sep=",",
            mode=("a" if append else "w"),
            header=not (append and os.path.isfile(fpath)),
            index=True,
            index_label="timestamp",
            encoding="utf-8",
        )
        # Write the full set of metrics to a JSON file.
        jdump = json.dumps({timestamp: scores})[1:-1]  # bracket-less dict
        fpath = os.path.join(self.folder, f"{prefix}.json")
        mode = "a" if append and os.path.isfile(fpath) else "w"
        with open(fpath, mode=mode, encoding="utf-8") as file:
            # First time, initialize the json file as a dict.
            if mode == "w":
                file.write(f"{{\n{jdump}\n}}")
            # Otherwise, append the record into the existing json dict.
            else:
                file.truncate(file.tell() - 2)  # remove trailing "\n}"
                file.write(f",\n{jdump}\n}}")  # append, then restore "\n}"
        # Return the timestamp label.
        return timestamp

    def checkpoint(
        self,
        model: Optional[Model] = None,
        optimizer: Optional[Optimizer] = None,
        metrics: Optional[Dict[str, Union[float, np.ndarray]]] = None,
        first_call: bool = False,
    ) -> str:
        """Checkpoint inputs, using a common timestamp label.

        Parameters
        ----------
        model: Model or None, default=None
            Optional Model to checkpoint.
            This will call `self.save_model(config=False, state=True)`.
        optimizer: Optimizer or None, default=None
            Optional Optimizer to checkpoint.
            This will call `self.save_optimize(config=False, state=True)`.
        metrics: dict[str, (float | np.ndarray)] or None, default=None
            Optional dict of metrics to checkpoint.
            This will call `self.save_metrics(append=True)`.
        first_call: bool, default=False
            Flag indicating whether to treat this checkpoint as the first
            one. If True, export the model and optimizer configurations
            and/or erase pre-existing configuration and metrics files.

        Returns
        -------
        timestamp: str
            Timestamp string labeling the model weights and optimizer state
            files, as well as the values appended to the metrics files.
        """
        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        remove = []  # type: List[str]
        if model:
            self.save_model(
                model, config=first_call, state=True, timestamp=timestamp
            )
        elif first_call:
            remove.append(os.path.join(self.folder, "model_config.json"))
        if optimizer:
            self.save_optimizer(
                optimizer, config=first_call, state=True, timestamp=timestamp
            )
        elif first_call:
            remove.append(os.path.join(self.folder, "optimizer_config.json"))
        if metrics:
            append = not first_call
            self.save_metrics(
                metrics, prefix="metrics", append=append, timestamp=timestamp
            )
        elif first_call:
            remove.append(os.path.join(self.folder, "metrics.csv"))
            remove.append(os.path.join(self.folder, "metrics.json"))
        for path in remove:
            if os.path.isfile(path):
                os.remove(path)
        return timestamp

    # Loading methods

    def load_model(
        self,
        model: Optional[Model] = None,
        timestamp: Optional[str] = None,
        load_state: bool = True,
    ) -> Model:
        """Instantiate a Model and/or reset its weights from a save file.

        Parameters
        ----------
        model: Model or None, default=None
            Optional Model, the weights of which to reload.
            If None, instantiate from the model config file (or raise).
        timestamp: str or None, default=None
            Optional timestamp string labeling the weights to reload.
            If None, use the weights with the most recent timestamp.
        load_state: bool, default=True
            Flag specifying whether model weights are to be reloaded.
            If `False`, `timestamp` will be ignored.
        """
        # Type-check or reload the Model from a config file.
        if model is None:
            fpath = os.path.join(self.folder, "model_config.json")
            if not os.path.isfile(fpath):
                raise FileNotFoundError(
                    "Cannot reload Model: config file not found."
                )
            model = deserialize_object(fpath)  # type: ignore
            if not isinstance(model, Model):
                raise TypeError(
                    f"The object reloaded from {fpath} is not a Model."
                )
        if not isinstance(model, Model):
            raise TypeError("'model' should be a Model or None.")
        # Load the model weights and assign them.
        if load_state:
            weights = self._load_state("model", timestamp=timestamp)
            model.set_weights(weights)
        return model

    def load_optimizer(
        self,
        optimizer: Optional[Optimizer] = None,
        timestamp: Optional[str] = None,
        load_state: bool = True,
    ) -> Optimizer:
        """Instantiate an Optimizer and/or reset its state from a save file.

        Parameters
        ----------
        optimizer: Optimizer or None, default=None
            Optional Optimizer, the weights of which to reload.
            If None, instantiate from the optimizer config file (or raise).
        timestamp: str or None, default=None
            Optional timestamp string labeling the state to reload.
            If None, use the state with the most recent timestamp.
        load_state: bool, default=True
            Flag specifying whether optimizer state are to be reloaded.
            If `False`, `timestamp` will be ignored.
        """
        # Type-check or reload the Optimizer from a config file.
        if optimizer is None:
            fpath = os.path.join(self.folder, "optimizer_config.json")
            if not os.path.isfile(fpath):
                raise FileNotFoundError(
                    "Cannot reload Optimizer: config file not found."
                )
            config = json_load(fpath)
            optimizer = Optimizer.from_config(config)
        if not isinstance(optimizer, Optimizer):
            raise TypeError("'optimizer' should be an Optimizer or None.")
        # Load the optimizer state and assign it.
        if load_state:
            state = self._load_state("optimizer", timestamp=timestamp)
            optimizer.set_state(state)
        return optimizer

    def _load_state(
        self,
        prefix: str,
        timestamp: Optional[str] = None,
    ) -> Any:
        """Reload data from a state checkpoint file.

        Parameters
        ----------
        prefix: str
            Prefix to the target state file.
        timestamp: str or None, default=None
            Optional timestamp string labeling the state to reload.
            If None, use the state with the most recent timestamp.
        """
        if isinstance(timestamp, str):
            fname = f"{prefix}_state_{timestamp}.json"
        else:
            files = self.sort_matching_files(f"{prefix}_state")
            if not files:
                raise FileNotFoundError(
                    f"Cannot reload {prefix} state: no state file found."
                )
            fname = files[-1]
        return json_load(os.path.join(self.folder, fname))

    def load_metrics(
        self,
        prefix: str = "metrics",
        timestamp: Optional[str] = None,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """Reload checkpointed metrics.

        To only reload scalar metrics as a timestamp-indexed dataframe,
        see the `load_scalar_metrics` method.

        Parameters
        ----------
        prefix: str, default="metrics"
            Prefix to the metrics save file's name.
        timestamp: str or None, default=None
            Optional timestamp string labeling the metrics to reload.
            If None, return all checkpointed metrics.

        Returns
        -------
        metrics: dict[str, dict[str, (float | np.ndarray)]]
            Dict of metrics, with `{timestamp: {key: value}}` format.
            If the `timestamp` argument was not None, the first dimension
            will only contain one key, which is that timestamp.
        """
        fpath = os.path.join(self.folder, f"{prefix}.json")
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Cannot reload metrics: file {fpath} does not exit."
            )
        with open(fpath, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        if timestamp:
            if timestamp not in metrics:
                raise KeyError(
                    f"The reloaded metrics have no {timestamp}-labeled entry."
                )
            metrics = {timestamp: metrics[timestamp]}
        return {
            timestamp: {
                key: np.array(val) if isinstance(val, list) else val
                for key, val in scores.items()
            }
            for timestamp, scores in metrics.items()
        }

    def load_scalar_metrics(
        self,
        prefix: str = "metrics",
    ) -> pd.DataFrame:
        """Return a pandas DataFrame storing checkpointed scalar metrics.

        To reload all checkpointed metrics (i.e. scalar and numpy array ones)
        see the `load_metrics` method.

        Parameters
        ----------
        prefix: str, default="metrics"
            Prefix to the metrics save file's name.

        Returns
        -------
        metrics: pandas.DataFrame
            DataFrame storing timestamp-indexed scalar metrics.
        """
        fpath = os.path.join(self.folder, f"{prefix}.csv")
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Cannot reload scalar metrics: file {fpath} does not exit."
            )
        return pd.read_csv(fpath, index_col="timestamp")
