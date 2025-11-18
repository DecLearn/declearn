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

"""Load the gRPC backend code auto-generated from "message.proto".

Instructions to re-generate the code:

* From the commandline, with 'protobufs' as working directory, run:
    - `python -m grpc_tools.protoc -I . --python_out=. \
              --grpc_python_out=. ./message.proto`
    - `sed -i -E 's/^import.*_pb2/from . \0/' ./*_pb2*.py`

* On MAC OS, replace the second command with:
    - `sed -i '' -E 's/^(import.*_pb2)/from . \1/' ./*_pb2*.py`
"""

try:
    from . import message_pb2, message_pb2_grpc
except ImportError as err:
    raise ImportError(
        "Failed to import grpc protobuf code. Try re-generating the files?\n"
        "(Refer to 'declearn.communication.grpc.protobufs.__init__.py'.)"
    ) from err
