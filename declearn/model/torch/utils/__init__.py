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

"""Utils for torch backend support code.

GPU/CPU backing device management utils
---------------------------------------
* [AutoDeviceModule][declearn.model.torch.utils.AutoDeviceModule]:
    Wrapper for a `torch.nn.Module`, automating device-management.
* [select_device][declearn.model.torch.utils.select_device]:
    Select a backing device to use based on inputs and availability.
"""

from ._gpu import AutoDeviceModule, select_device
