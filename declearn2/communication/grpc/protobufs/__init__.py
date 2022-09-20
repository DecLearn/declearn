# coding: utf-8

"""Load the gRPC backend code auto-generated from "message.proto".

Instructions to re-generate the code:
* From the commandline, with 'protobufs' as working directory, run:
  python -m grpc_tools.protoc -I . --python_out=. \
         --grpc_python_out=. ./message.proto
  sed -i -E 's/^import.*_pb2/from . \0/' ./*_pb2*.py
* On MAC OS, replace the second command with:
  sed -i '' -E 's/^(import.*_pb2)/from . \1/' ./*_pb2*.py
"""

try:
    from . import message_pb2
    from . import message_pb2_grpc
except ImportError as err:
    raise ImportError(
        "Failed to import grpc protobuf code. Try re-generating the files?\n"
        "(Refer to 'declearn2.communication.grpc.protobufs.__init__.py'.)"
    ) from err
