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

"""Backend tools for the network communication layer of DecLearn.

This submodule provides with a variety of components that are used
to build the shared backend of network communication endpoints, in
a framework-agnostic way.

These tools are mostly hard-coded, they are not meant to be of any
use for end-users, nor to be exposed by any API public method. The
only exception is that `NetworkServer` uses a `handler` attribute
that is an instantiated `MessagesHandler`, which protocol-specific
subclasses may need to access.

Server-side servicer for message-passing
----------------------------------------

* [MessagesHandler][declearn.communication.api.backend.MessagesHandler]
    Minimal protocol-agnostic server-side messages handler

Submodules exposing truly-backend bricks
----------------------------------------

* [actions][declearn.communication.api.backend.actions]:
    Fundamental backend hard-coded message containers for DecLearn.
* [flags][declearn.communication.api.backend.flags]:
    Communication flags used by the declearn communication backend.
"""

from . import actions, flags
from ._handler import MessagesHandler
