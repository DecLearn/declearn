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

"""Communication flags used by the declearn communication backend.

This module exposes conventional flags, which are all str constants.
"""

from declearn.version import VERSION

__all__ = [
    "CHECK_MESSAGE_TIMEOUT",
    "INVALID_MESSAGE",
    "REGISTERED_ALREADY",
    "REGISTERED_WELCOME",
    "REGISTRATION_CLOSED",
    "REGISTRATION_OPEN",
    "REGISTRATION_UNSTARTED",
    "REJECT_INCOMPATIBLE_VERSION",
    "REJECT_UNREGISTERED",
    "REJECT_UNREGISTERED_CHUNKED",
]


# Registration flags.
REGISTERED_ALREADY = "you were already registered"
REGISTERED_WELCOME = "welcome, you are now registered"
REGISTRATION_CLOSED = "registration is closed"
REGISTRATION_OPEN = "registration is open"
REGISTRATION_UNSTARTED = "registration is not opened yet"

# Error flags.
CHECK_MESSAGE_TIMEOUT = "no available message at timeout"
INVALID_MESSAGE = "invalid message"
REJECT_INCOMPATIBLE_VERSION = (
    "cannot register due to the DecLearn version in use; "
    f"please update to `declearn ~= {VERSION}`"
)
REJECT_UNREGISTERED = "rejected: not a registered user"
REJECT_UNREGISTERED_CHUNKED = (
    "chunked messages from unregistered clients are not allowed"
)
