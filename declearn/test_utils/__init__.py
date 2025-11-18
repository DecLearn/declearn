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

"""Collection of utils for running tests and examples around declearn.

This submodule is not imported with declearn by default - it requires
being explicitly imported, and should not be so by end-users, unless
they accept the risk of using unstable features.

This submodule is *not* considered part of the stable declearn API,
meaning that its contents may change without warnings. Its features
are not designed to be used outside of the scope of declearn-shipped
tests and examples. It may also serve to introduce experimental new
features that may be ported to the stable API in the future.
"""

from ._argparse import setup_client_argparse, setup_server_argparse
from ._assertions import (
    assert_batch_equal,
    assert_dict_equal,
    assert_json_serializable_dict,
    assert_list_equal,
)
from ._convert import to_numpy
from ._gen_ssl import generate_ssl_certificates
from ._imports import make_importable
from ._network import (
    MockNetworkClient,
    MockNetworkServer,
    setup_mock_network_endpoints,
)
from ._secagg import build_secagg_controllers
from ._vectors import (
    FrameworkType,
    GradientsTestCase,
    list_available_frameworks,
)
