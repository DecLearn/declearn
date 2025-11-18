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

"""Simple Torch-backed CNN model for the MNIST quickrun example."""

import torch

from declearn.model.torch import TorchModel


stack = [
    torch.nn.Unflatten(dim=0, unflattened_size=(-1, 1)),
    torch.nn.Conv2d(1, 8, 3, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Dropout(0.25),
    torch.nn.Flatten(),
    torch.nn.Linear(1352, 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(64, 10),
    torch.nn.Softmax(dim=-1),
]
network = torch.nn.Sequential(*stack)

# This needs to be called "model"; otherwise, a different name must be
# specified via the experiment's TOML configuration file.
model = TorchModel(network, loss=torch.nn.CrossEntropyLoss())
