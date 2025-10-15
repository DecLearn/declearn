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

"""Functional test of the declearn quickrun example."""

import os
import pathlib

import numpy as np
import pytest

from declearn.dataset import split_data
from declearn.quickrun import quickrun

MODEL_CODE = """
from declearn.model.sklearn import SklearnSGDModel

model = SklearnSGDModel.from_parameters(kind="classifier", penalty="none")
"""

CONFIG_TOML = """
[network]
protocol = "websockets"
host = "127.0.0.1"
port = 8080
heartbeat = 0.1

[data]

[optim]
[optim.client_opt]
lrate = 0.01
modules = ["adam"]
regularizers = ["lasso"]

[run]
rounds = 2
[run.register]
min_clients = 2
[run.training]
batch_size = 48
n_steps = 100
[run.evaluate]
batch_size = 128

[experiment]
metrics = [
    ["multi-classif", {labels = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9]}]
]
"""


@pytest.mark.asyncio
async def test_quickrun_mnist(tmp_path: str) -> None:
    """Run a very basic MNIST example using 'declearn-quickrun'."""
    # Download, prepare and split the MNIST dataset into iid shards.
    split_data(tmp_path, n_shards=2, seed=0)
    # Flatten out the input images to enable their processing with sklearn.
    for path in pathlib.Path(tmp_path).glob("data_iid/client_*/*_data.npy"):
        images = np.load(path)
        np.save(path, images.reshape((-1, 28 * 28)))
    # Write down a very basic TOML config and python model files.
    model = os.path.join(tmp_path, "model.py")
    with open(model, "w", encoding="utf-8") as file:
        file.write(MODEL_CODE)
    config = os.path.join(tmp_path, "config.toml")
    with open(config, "w", encoding="utf-8") as file:
        file.write(CONFIG_TOML)
    # Run the quickrun experiment.
    await quickrun(config)
