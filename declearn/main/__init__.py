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

"""Main classes implementing a Federated Learning process.

This module mainly implements the following two classes:

* [FederatedClient][declearn.main.FederatedClient]:
    Client-side main Federated Learning orchestrating class
* [FederatedServer][declearn.main.FederatedServer]:
    Server-side main Federated Learning orchestrating class

This module also implements the following submodules, used by the former:

* [config][declearn.main.config]:
    Server-side dataclasses that specify a FL process's parameter.
    The main classes implemented here are `FLRunConfig` and `FLOptimConfig`,
    that implement parameters' parsing from python objets or from TOML files.
* [utils][declearn.main.utils]:
    Various utils to the FL process.
    The main class of interest for end-users is `TrainingManager`, that
    implements client-side training and evaluation routines, and may
    therefore be leveraged in a non-FL setting or to implement other
    FL process routines than the centralized one defined here.

Finally, the [privacy][declearn.main.privacy] submodule is DEPRECATED as of
DecLearn 2.6 and will be removed in DecLearn 2.8. It was moved and renamed
to [declearn.training.dp][]. It can still be manually imported under its
deprecated name (containing re-exports of moved contents).
"""

from . import config, utils
from ._client import FederatedClient
from ._server import FederatedServer
