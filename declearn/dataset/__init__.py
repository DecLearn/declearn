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

"""Dataset-interface API and actual implementations module.

A 'Dataset'  is an interface towards data that exposes methods to query batched
data  samples and key metadata while remaining agnostic of the way the data is
actually being loaded (from a source file, a database, another API...).

This declearn submodule provides with:
* Dataset : abstract class defining an API to access training or testing data
* InMemoryDataset : Dataset subclass serving numpy(-like) memory-loaded data
arrays
"""

from ._base import Dataset, DataSpecs, load_dataset_from_json

# from ._sparse import sparse_from_file, sparse_to_file
from ._inmemory import InMemoryDataset
