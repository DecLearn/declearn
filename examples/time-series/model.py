# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Script that creates the model for the Hand Poses sEMG timeseries example."""

from dataclasses import astuple, dataclass

import torch
from torch import nn

__all__ = ["SimpleMaskedTSAutoEncoder"]


# wrap the model config into a dataclass for readability.
@dataclass
class ModelConfigsInput:
    """Container for the model input.

    Fields
    ------
    input_dim: int
        Input dimension for the first layer. Default to 50
    hidden_dim: int
        Dimension of the hidden layer (latent dimension). Default to 30
    mask_ratio: float
        Percentage of input masking rate. Default to 25%.
    """

    input_dim: int = 50
    hidden_dim: int = 30
    mask_ratio: float = 0.25


class SimpleMaskedTSAutoEncoder(nn.Module):
    """
    A simple implementation of a masked auto-encoder for time-series.

    Attributes
    ----------
    input_dim: int
        Input dimension for the first layer.
    mask_ratio: float
        Percentage of input masking rate. Default to 25%.
    encoder: nn.Sequential
        Torch.nn.Sequential instance representing the architecture of the
        encoder.
    decoder: nn.Sequential
        Torch.nn.Sequential instance representing the architecture of the
        decoder.
    """

    def __init__(self, model_configs_input: ModelConfigsInput):
        """Instanciates a Masked auto-encoder model for training.

        Parameters
        ----------
        model_configs_input: ModelConfigsInput
            User input for the model configuration.
        """
        super().__init__()

        input_dim, hidden_dim, mask_ratio = astuple(model_configs_input)

        self.input_dim = input_dim
        self.mask_ratio = mask_ratio
        # encoding time serie
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # decoding time serie
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Assumes input is normalized to [0,1]
        )

    def forward(self, x):
        mask = self._create_mask(x.shape)
        masked_x = x * mask

        encoded = self.encoder(masked_x)
        decoded = self.decoder(encoded)

        # NOTE for now do not return tuple (decoded, mask) since it's not
        # necessary to compute the loss for now.
        return decoded

    def _create_mask(self, shape):
        # Randomly mask some positions
        mask = torch.rand(shape) > self.mask_ratio
        return mask.float()
