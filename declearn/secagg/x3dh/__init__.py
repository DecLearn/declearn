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

"""Extended Triple Diffie-Hellman (X3DH) key agreement tools.

This module provides with an implementation of the X3DH protocol [1],
that enables setting up pairwise ephemeral shared secret keys across
a network of peers based on a pre-existing public key infrastructure.

Setup routines
--------------

* [run_x3dh_setup_client][declearn.secagg.x3dh.run_x3dh_setup_client]:
    Participate in a X3DH (Extended Triple Diffie-Hellman) protocol.
* [run_x3dh_setup_server][declearn.secagg.x3dh.run_x3dh_setup_server]:
    Orchestrate a X3DH (Extended Triple Diffie-Hellman) protocol run.

Backend
-------

* [X3DHManager][declearn.secagg.x3dh.X3DHManager]:
    X3DH (Extended Triple Diffie-Hellman) key agreement manager.
* [messages][declearn.secagg.x3dh.messages]:
    Submodule providing with dedicated messages used for X3DH setup.

References
----------
[1] Marlinspike & Perrin, 2016.
    The X3DH Key Agreement Protocol.
    https://www.signal.org/docs/specifications/x3dh/
"""

from . import messages
from ._setup import run_x3dh_setup_client, run_x3dh_setup_server
from ._x3dh import X3DHManager
