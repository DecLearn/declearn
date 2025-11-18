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

"""Unit tests for Checkpointer class."""

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Union
from unittest import mock

import numpy as np
import pandas as pd  # type: ignore
import pytest
from sklearn.linear_model import SGDClassifier  # type: ignore

from declearn.main.utils import Checkpointer
from declearn.model.api import Model
from declearn.model.sklearn import SklearnSGDModel
from declearn.optimizer import Optimizer
from declearn.utils import json_load

# Fixtures and utils


@pytest.fixture(name="checkpointer")
def fixture_checkpointer(tmp_path) -> Iterator[Checkpointer]:
    """Create a checkpointer within a temp dir"""
    yield Checkpointer(tmp_path, 2)


@pytest.fixture(name="model")
def fixture_model() -> SklearnSGDModel:
    """Crete a toy binary-classification model."""
    model = SklearnSGDModel(SGDClassifier())
    model.initialize({"features_shape": (8,), "classes": np.arange(2)})
    return model


@pytest.fixture(name="optimizer")
def fixture_optimizer() -> Optimizer:
    """Create a toy optimizer"""
    testopt = Optimizer(lrate=1.0, modules=[("momentum", {"beta": 0.95})])
    return testopt


@pytest.fixture(name="metrics")
def fixture_metrics() -> Dict[str, float]:
    """Create a metrics fixture"""
    return {"loss": 0.5}


def create_state_files(folder: str, type_obj: str, n_files: int) -> List[str]:
    """Create test state files in checkpointer.ckpt"""
    files = [
        f"{type_obj}_state_23-01-{21 + idx}_15-45-35.json"
        for idx in range(n_files)
    ]
    for name in files:
        with open(os.path.join(folder, name), "w", encoding="utf-8") as file:
            json.dump({"test": "state"}, file)
    return files


def create_config_file(checkpointer: Checkpointer, type_obj: str) -> str:
    """Create test cfg files in checkpointer.ckpt"""
    path = os.path.join(checkpointer.folder, f"{type_obj}_config.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"test": "config"}, file)
    return f"{type_obj}_config.json"


# Actual tests


class TestCheckpointer:
    """Unit tests for Checkpointer class."""

    def test_init_default(self, tmp_path: str) -> None:
        """Test `Checkpointer.__init__` with `max_history=None`."""
        checkpointer = Checkpointer(folder=tmp_path, max_history=None)
        assert checkpointer.folder == tmp_path
        assert Path(checkpointer.folder).is_dir()
        assert checkpointer.max_history is None

    def test_init_max_history(self, tmp_path: str) -> None:
        """Test `Checkpointer.__init__` with `max_history=2`."""
        checkpointer = Checkpointer(folder=tmp_path, max_history=2)
        assert checkpointer.folder == tmp_path
        assert Path(checkpointer.folder).is_dir()
        assert checkpointer.max_history == 2

    def test_init_fails(self, tmp_path: str) -> None:
        """Test `Checkpointer.__init__` raises on negative `max_history`."""
        with pytest.raises(TypeError):
            Checkpointer(folder=tmp_path, max_history=-1)

    def test_from_specs(self, tmp_path: str) -> None:
        """Test that `Checkpointer.from_specs` works properly.

        This test is multi-part rather than unitary as the method
        is merely boilerplate code refactored into a classmethod.
        """
        tmp_path = str(tmp_path)  # note: PosixPath
        specs_list = [
            tmp_path,
            {"folder": tmp_path, "max_history": None},
            Checkpointer(tmp_path),
        ]
        # Iteratively test the various types of acceptable specs.
        for specs in specs_list:
            ckpt = Checkpointer.from_specs(specs)  # type: ignore
            assert isinstance(ckpt, Checkpointer)
            assert ckpt.folder == tmp_path
            assert ckpt.max_history is None
        # Also test that the documented TypeError is raised.
        with pytest.raises(TypeError):
            Checkpointer.from_specs(0)  # type: ignore

    def test_garbage_collect(self, tmp_path: str) -> None:
        """Test `Checkpointer.garbage_collect` when collection is needed."""
        # Set up a checkpointer with max_history=2 and 3 state files.
        checkpointer = Checkpointer(folder=tmp_path, max_history=2)
        names = sorted(create_state_files(tmp_path, "model", n_files=3))
        checkpointer.garbage_collect("model_state")
        # Verify that the "oldest" file was removed.
        files = sorted(os.listdir(checkpointer.folder))
        assert len(files) == checkpointer.max_history
        assert files == names[1:]  # i.e. [-max_history:]

    def test_garbage_collect_no_collection(self, tmp_path: str) -> None:
        """Test `Checkpointer.garbage_collect` when collection is unneeded."""
        # Set up a checkpointer with max_history=3 and 2 state files.
        checkpointer = Checkpointer(folder=tmp_path, max_history=3)
        names = sorted(create_state_files(tmp_path, "model", n_files=2))
        checkpointer.garbage_collect("model_state")
        # Verify that no files were removed.
        files = sorted(os.listdir(checkpointer.folder))
        assert files == names

    def test_garbage_collect_infinite_history(self, tmp_path: str) -> None:
        """Test `Checkpointer.garbage_collect` when `max_history=None`."""
        # Set up a checkpointer with max_history=None and 3 state files.
        checkpointer = Checkpointer(folder=tmp_path, max_history=None)
        names = sorted(create_state_files(tmp_path, "model", n_files=3))
        checkpointer.garbage_collect("model_state")
        # Verify that no files were removed.
        files = sorted(os.listdir(checkpointer.folder))
        assert files == names

    def test_sort_matching_files(self, tmp_path: str) -> None:
        """Test `Checkpointer.sort_matching_files`."""
        checkpointer = Checkpointer(folder=tmp_path)
        names = sorted(create_state_files(tmp_path, "model", n_files=3))
        create_state_files(tmp_path, "optimizer", n_files=2)
        files = checkpointer.sort_matching_files("model_state")
        assert names == files

    @pytest.mark.parametrize("state", [True, False], ids=["state", "no_state"])
    @pytest.mark.parametrize("config", [True, False], ids=["config", "no_cfg"])
    def test_save_model(
        self, tmp_path: str, model: Model, config: bool, state: bool
    ) -> None:
        """Test `Checkpointer.save_model` with provided parameters."""
        checkpointer = Checkpointer(folder=tmp_path)
        timestamp = checkpointer.save_model(model, config, state)
        # Verify config save file's existence.
        cfg_path = os.path.join(checkpointer.folder, "model_config.json")
        if config:
            assert Path(cfg_path).is_file()
        else:
            assert not Path(cfg_path).is_file()
        # Vertify weights save file's existence.
        if state:  # test state file save
            assert isinstance(timestamp, str)
            state_path = os.path.join(
                checkpointer.folder, f"model_state_{timestamp}.json"
            )
            assert Path(state_path).is_file()
        else:
            assert timestamp is None
            assert not checkpointer.sort_matching_files("model_state")

    @pytest.mark.parametrize("state", [True, False], ids=["state", "no_state"])
    @pytest.mark.parametrize("config", [True, False], ids=["config", "no_cfg"])
    def test_save_optimizer(
        self, tmp_path: str, optimizer: Optimizer, config: bool, state: bool
    ) -> None:
        """Test `Checkpointer.save_optimizer` with provided parameters."""
        checkpointer = Checkpointer(folder=tmp_path)
        timestamp = checkpointer.save_optimizer(optimizer, config, state)
        # Verify config save file's existence.
        cfg_path = os.path.join(checkpointer.folder, "optimizer_config.json")
        if config:
            assert Path(cfg_path).is_file()
        else:
            assert not Path(cfg_path).is_file()
        # Vertify state save file's existence.
        if state:
            assert isinstance(timestamp, str)
            state_path = os.path.join(
                checkpointer.folder, f"optimizer_state_{timestamp}.json"
            )
            assert Path(state_path).is_file()
        else:
            assert timestamp is None
            assert not checkpointer.sort_matching_files("optimizer_state")

    def test_save_metrics(self, tmp_path: str) -> None:
        """Test that `Checkpointer.save_metrics` works as expected.

        This is a multi-part test rather than unit one, to verify
        that the `append` parameter and its backend work properly.
        """
        # Setup for this multi-part test.
        metrics: Dict[str, Union[float, np.ndarray]] = {
            "foo": 42.0,
            "bar": np.array([0, 1]),
        }
        checkpointer = Checkpointer(tmp_path)
        csv_path = os.path.join(tmp_path, "metrics.csv")
        json_path = os.path.join(tmp_path, "metrics.json")

        # Case 'append=True' but the files do not exist.
        checkpointer.save_metrics(metrics, append=True, timestamp="0")
        assert os.path.isfile(csv_path)
        assert os.path.isfile(json_path)
        scalars = pd.DataFrame({"timestamp": [0], "foo": [42.0]})
        assert (pd.read_csv(csv_path) == scalars).all(axis=None)
        m_json = {"foo": 42.0, "bar": [0, 1]}
        assert json_load(json_path) == {"0": m_json}

        # Case 'append=False', overwriting existing files.
        checkpointer.save_metrics(metrics, append=False, timestamp="0")
        assert (pd.read_csv(csv_path) == scalars).all(axis=None)
        assert json_load(json_path) == {"0": m_json}

        # Case 'append=True', appending to existing files.
        checkpointer.save_metrics(metrics, append=True, timestamp="1")
        scalars = pd.DataFrame({"timestamp": [0, 1], "foo": [42.0, 42.0]})
        m_json = {"0": m_json, "1": m_json}
        assert (pd.read_csv(csv_path) == scalars).all(axis=None)
        assert json_load(json_path) == m_json

    @pytest.mark.parametrize("first", [True, False], ids=["first", "notfirst"])
    def test_checkpoint(
        self, tmp_path: str, model: Model, optimizer: Optimizer, first: bool
    ) -> None:
        """Test that `Checkpointer.checkpoint` works as expected."""
        # Set up a checkpointer and call its checkpoint method.
        checkpointer = Checkpointer(tmp_path)
        metrics = {"foo": 42.0, "bar": np.array([0, 1])}
        if first:  # create some files that should be removed on `first_call`
            create_config_file(checkpointer, "model")
        timestamp = checkpointer.checkpoint(
            model=model,
            optimizer=optimizer,
            metrics=metrics,  # type: ignore
            first_call=first,
        )
        assert isinstance(timestamp, str)
        # Verify whether config and metric files exist, as expected.
        m_cfg = os.path.join(tmp_path, "model_config.json")
        o_cfg = os.path.join(tmp_path, "optimizer_config.json")
        if first:
            assert os.path.isfile(m_cfg)
            assert os.path.isfile(o_cfg)
        else:
            assert not os.path.isfile(m_cfg)
            assert not os.path.isfile(o_cfg)
        # Verify that state and metric files exist as expected.
        path = os.path.join(tmp_path, f"model_state_{timestamp}.json")
        assert os.path.isfile(path)
        path = os.path.join(tmp_path, f"optimizer_state_{timestamp}.json")
        assert os.path.isfile(path)
        assert os.path.isfile(os.path.join(tmp_path, "metrics.csv"))
        assert os.path.isfile(os.path.join(tmp_path, "metrics.json"))

    @pytest.mark.parametrize("state", [True, False], ids=["state", "no_state"])
    @pytest.mark.parametrize("config", [True, False], ids=["config", "model"])
    def test_load_model(
        self, tmp_path: str, model: Model, config: bool, state: bool
    ) -> None:
        """Test `Checkpointer.load_model` with provided parameters."""
        checkpointer = Checkpointer(tmp_path)
        # Save the model (config + weights), then reload based on parameters.
        timestamp = checkpointer.save_model(model, config=True, state=True)
        with mock.patch.object(type(model), "set_weights") as p_set_weights:
            loaded_model = checkpointer.load_model(
                model=(None if config else model),
                timestamp=(timestamp if config else None),  # arbitrary swap
                load_state=state,
            )
        # Verify that the loadd model is either the input one or similar.
        if config:
            assert isinstance(loaded_model, type(model))
            assert loaded_model is not model
            assert loaded_model.get_config() == model.get_config()
        else:
            assert loaded_model is model
        # Verify that `set_weights` was called, with proper values.
        if state:
            p_set_weights.assert_called_once()
            if config:
                assert loaded_model.get_weights() == model.get_weights()
        else:
            p_set_weights.assert_not_called()

    def test_load_model_fails(self, tmp_path: str, model: Model) -> None:
        """Test that `Checkpointer.load_model` raises excepted errors."""
        checkpointer = Checkpointer(tmp_path)
        # Case when the weights file is missing.
        checkpointer.save_model(model, config=False, state=False)
        with pytest.raises(FileNotFoundError):
            checkpointer.load_model(model=model, load_state=True)
        # Case when the config file is mising.
        checkpointer.save_model(model, config=False, state=True)
        with pytest.raises(FileNotFoundError):
            checkpointer.load_model(model=None)
        # Case when a wrong model input is provided.
        with pytest.raises(TypeError):
            checkpointer.load_model(model="wrong-type")  # type: ignore

    @pytest.mark.parametrize("state", [True, False], ids=["state", "no_state"])
    @pytest.mark.parametrize("config", [True, False], ids=["config", "optim"])
    def test_load_optimizer(
        self, tmp_path: str, optimizer: Optimizer, config: bool, state: bool
    ) -> None:
        """Test `Checkpointer.load_optimizer` with provided parameters."""
        checkpointer = Checkpointer(tmp_path)
        # Save the optimizer (config + state), then reload based on parameters.
        stamp = checkpointer.save_optimizer(optimizer, config=True, state=True)
        with mock.patch.object(Optimizer, "set_state") as p_set_state:
            loaded_optim = checkpointer.load_optimizer(
                optimizer=(None if config else optimizer),
                timestamp=(stamp if config else None),  # arbitrary swap
                load_state=state,
            )
        # Verify that the loaded optimizer is either the input one or similar.
        if config:
            assert isinstance(loaded_optim, Optimizer)
            assert loaded_optim is not optimizer
            assert loaded_optim.get_config() == optimizer.get_config()
        else:
            assert loaded_optim is optimizer
        # Verify that `set_state` was called, with proper values.
        if state:
            p_set_state.assert_called_once()
            if config:
                assert loaded_optim.get_state() == optimizer.get_state()
        else:
            p_set_state.assert_not_called()

    def test_load_optimizer_fails(
        self, tmp_path: str, optimizer: Optimizer
    ) -> None:
        """Test that `Checkpointer.load_optimizer` raises excepted errors."""
        checkpointer = Checkpointer(tmp_path)
        # Case when the state file is missing.
        checkpointer.save_optimizer(optimizer, config=False, state=False)
        with pytest.raises(FileNotFoundError):
            checkpointer.load_optimizer(optimizer=optimizer, load_state=True)
        # Case when the config file is mising.
        checkpointer.save_optimizer(optimizer, config=False, state=True)
        with pytest.raises(FileNotFoundError):
            checkpointer.load_optimizer(optimizer=None)
        # Case when a wrong optimizer input is provided.
        with pytest.raises(TypeError):
            checkpointer.load_optimizer(optimizer="wrong-type")  # type: ignore

    def test_load_metrics(self, tmp_path: str) -> None:
        """Test that `Checkpointer.load_metrics` works properly."""
        # Setup things by saving a couple of sets of metrics.
        metrics: Dict[str, Union[float, np.ndarray]] = {
            "foo": 42.0,
            "bar": np.array([0, 1]),
        }
        checkpointer = Checkpointer(tmp_path)
        time_0 = checkpointer.save_metrics(metrics, append=False)
        time_1 = checkpointer.save_metrics(metrics, append=True)
        # Test reloading all checkpointed metrics.
        reloaded = checkpointer.load_metrics(timestamp=None)
        assert isinstance(reloaded, dict)
        assert reloaded.keys() == {time_0, time_1}
        for scores in reloaded.values():
            assert isinstance(scores, dict)
            assert scores.keys() == metrics.keys()
            assert scores["foo"] == metrics["foo"]
            assert (scores["bar"] == metrics["bar"]).all()  # type: ignore
        # Test reloading only metrics from one timestamp.
        reloaded = checkpointer.load_metrics(timestamp=time_0)
        assert isinstance(reloaded, dict)
        assert reloaded.keys() == {time_0}

    def test_load_scalar_metrics(self, tmp_path: str) -> None:
        """Test that `Checkpointer.load_scalar_metrics` works properly."""
        # Setup things by saving a couple of sets of metrics.
        metrics: Dict[str, Union[float, np.ndarray]] = {
            "foo": 42.0,
            "bar": np.array([0, 1]),
        }
        checkpointer = Checkpointer(tmp_path)
        time_0 = checkpointer.save_metrics(metrics, append=False)
        time_1 = checkpointer.save_metrics(metrics, append=True)
        expect = pd.DataFrame(
            {"foo": [42.0, 42.0], "timestamp": [time_0, time_1]}
        ).set_index("timestamp")
        # Test reloading scalar metrics.
        scores = checkpointer.load_scalar_metrics()
        assert isinstance(scores, pd.DataFrame)
        assert scores.index.names == expect.index.names
        assert scores.columns == expect.columns
        assert scores.shape == expect.shape
        assert (scores == expect).all(axis=None)
