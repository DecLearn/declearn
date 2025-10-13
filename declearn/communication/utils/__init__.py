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

"""Utils related to network communication endpoints' setup and usage.


Endpoints setup utils
---------------------

* [build_client][declearn.communication.utils.build_client]:
    Instantiate a NetworkClient, selecting its subclass based on protocol name.
* [build_server][declearn.communication.utils.build_server]:
    Instantiate a NetworkServer, selecting its subclass based on protocol name.
* [list_available_protocols]\
[declearn.communication.utils.list_available_protocols]:
    Return the list of readily-available network protocols.
* [NetworkClientConfig][declearn.communication.utils.NetworkClientConfig]:
    TOML-parsable dataclass for network clients' instantiation.
* [NetworkServerConfig][declearn.communication.utils.NetworkServerConfig]:
    TOML-parsable dataclass for network servers' instantiation.


Message-type control utils
--------------------------

* [ErrorMessageException][declearn.communication.utils.ErrorMessageException]:
    Exception raised when an unexpected 'Error' message is received.
* [MessageTypeException][declearn.communication.utils.MessageTypeException]:
    Exception raised when a received 'Message' has wrong type.
* [verify_client_messages_validity]\
[declearn.communication.utils.verify_client_messages_validity]:
    Verify that received serialized messages match an expected type.
* [verify_server_message_validity]\
[declearn.communication.utils.verify_server_message_validity]:
    Verify that a received serialized message matches expected type.
"""

from ._build import (
    _INSTALLABLE_BACKENDS,
    NetworkClientConfig,
    NetworkServerConfig,
    build_client,
    build_server,
    list_available_protocols,
)
from ._parse import (
    ErrorMessageException,
    MessageTypeException,
    verify_client_messages_validity,
    verify_server_message_validity,
)
