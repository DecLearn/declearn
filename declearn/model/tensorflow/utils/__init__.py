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

"""Utils for tensorflow backend support code.

GPU/CPU backing device management utils
---------------------------------------
* [move_layer_to_device]\
[declearn.model.tensorflow.utils.move_layer_to_device]:
    Create a copy of an input keras layer placed on a given device.
* [preserve_tensor_device]\
[declearn.model.tensorflow.utils.preserve_tensor_device]:
    Wrap a tensor-processing function to have it run on its inputs' device.
* [select_device]\
[declearn.model.tensorflow.utils.select_device]:
    Select a backing device to use based on inputs and availability.

Loss function management utils
------------------------------
* [build_keras_loss]\
[declearn.model.tensorflow.utils.build_keras_loss]:
    Type-check, deserialize and/or wrap a keras loss into a Loss object.

Better support for sparse tensor structures
-------------------------------------------
* [add_indexed_slices_support]\
[declearn.model.tensorflow.utils.add_indexed_slices_support]:
    Run a function on a pair of tensors, adding support for IndexedSlices.
"""

from ._gpu import (
    move_layer_to_device,
    preserve_tensor_device,
    select_device,
)
from ._loss import build_keras_loss
from ._slices import add_indexed_slices_support
