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

"""Dataset interface to wrap up 'torch.utils.data.Dataset' instances.

The main class implementing by this submodule is `TorchDataset`:

* [TorchDataset][declearn.dataset.torch.TorchDataset]:
    Dataset subclass serving torch Datasets.

Some utils are also exposed here, either used as part of the `TorchDataset`
backend or to be used in conjunction with it:

* [PoissonSampler][declearn.dataset.torch.PoissonSampler]:
    Custom `torch.utils.data.Sampler` implementing Poisson sampling.
* [collate_with_padding][declearn.dataset.torch.collate_with_padding]:
    Custom collate function that implements variable-lenght inputs' padding.
"""

from ._torch import TorchDataset
from ._utils import PoissonSampler, collate_with_padding
