# coding: utf-8

"""gRPC implementation of network communication endpoints."""

from . import protobufs
from ._client import GrpcClient
from ._server import GrpcServer

# Note: to generate protobufs/*.py files from the message.proto file,
# run the following instructions (with protobufs as working directory):
# python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. ./message.proto
# sed -i -E 's/^import.*_pb2/from . \0/' ./*_pb2*.py
# or for MAC OS: sed -i '' -E 's/^(import.*_pb2)/from . \1/' ./*_pb2*.py
