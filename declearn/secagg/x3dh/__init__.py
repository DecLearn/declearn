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

"""Extended Triple Diffie-Hellman (X3DH) key agreement tools.

* [X3DHManager][declearn.secagg.x3dh.X3DHManager]:
    X3DH (Extended Triple Diffie-Hellman) key agreement manager.
* [X3DHClientRound][declearn.secagg.x3dh.X3DHClientRound]:
    Client-side routine for X3DH (Extended Triple Diffie-Hellman) setup.
* [X3DHServerRound][declearn.secagg.x3dh.X3DHServerRound]:
    Server-side routine for X3DH (Extended Triple Diffie-Hellman) setup.
"""

from ._x3dh import X3DHManager
from ._routines import X3DHClientRound, X3DHServerRound
