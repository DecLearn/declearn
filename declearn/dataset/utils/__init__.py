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

"""Utils to manipulate datasets (load, save, split...).

Data loading and saving
-----------------------
declearn provides with utils to load and save array-like data tensors
to and from various file formats:

* [load_data_array][declearn.dataset.utils.load_data_array]:
    Load a data array (numpy, scipy, pandas) from a dump file.
* [save_data_array][declearn.dataset.utils.save_data_array]:
    Save a data array (numpy, scipy, pandas) to a dump file.
* [sparse_from_file][declearn.dataset.utils.sparse_from_file]:
    Backend to load a sparse matrix from a dump file.
* [sparse_to_file][declearn.dataset.utils.sparse_to_file]:
    Backend to save a sparse matrix to a dump file

Data splitting
--------------
* [split_multi_classif_dataset]
[declearn.dataset.utils.split_multi_classif_dataset]:
    Split a classification dataset into (opt. heterogeneous) shards.
"""

from ._save_load import load_data_array, save_data_array
from ._sparse import sparse_from_file, sparse_to_file
from ._split_classif import split_multi_classif_dataset
