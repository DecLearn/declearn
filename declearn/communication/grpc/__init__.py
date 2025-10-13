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

"""gRPC implementation of network communication endpoints.

The classes implemented here are:

* [GrpcClient][declearn.communication.grpc.GrpcClient]:
    Client-side network communication endpoint implementation using gRPC.
* [GrpcServer][declearn.communication.grpc.GrpcServer]:
    Server-side network communication endpoint implementation using gRPC.

The [protobufs][declearn.communication.grpc.protobufs] submodule is also
exposed, that provides with backend code auto-generated from a protobuf
file, and is not considered part of the declearn stable API.
"""

from . import protobufs
from ._client import GrpcClient
from ._server import GrpcServer
