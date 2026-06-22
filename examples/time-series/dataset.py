

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

"""Script to create MaskedAutoEncoder necessary Dataset."""

from torch.utils.data import Dataset


class MaskedAutoEncoderDataset(Dataset):
    def __init__(self, data):
        """
        Dataset for Time-series Masked Auto-encoder model example. 
        
        This dataset returns an input sample and a target derived from the same
        sample. It is typically used in self-supervised settings where the model
        learns to reconstruct missing or masked parts of the input.
        
        For now this class serves as a pytorch wrapper around the data and does not include the 
        transformation functionality.
        
        Args:
            data (list or array): Input data samples.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = sample
        return sample, label
