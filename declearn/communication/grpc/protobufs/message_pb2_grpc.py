# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import message_pb2 as message__pb2


class MessageBoardStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ping = channel.unary_unary(
                '/grpc.MessageBoard/ping',
                request_serializer=message__pb2.Empty.SerializeToString,
                response_deserializer=message__pb2.Empty.FromString,
                )
        self.send = channel.unary_stream(
                '/grpc.MessageBoard/send',
                request_serializer=message__pb2.Message.SerializeToString,
                response_deserializer=message__pb2.Message.FromString,
                )
        self.send_stream = channel.stream_stream(
                '/grpc.MessageBoard/send_stream',
                request_serializer=message__pb2.Message.SerializeToString,
                response_deserializer=message__pb2.Message.FromString,
                )


class MessageBoardServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ping(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def send(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def send_stream(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MessageBoardServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ping': grpc.unary_unary_rpc_method_handler(
                    servicer.ping,
                    request_deserializer=message__pb2.Empty.FromString,
                    response_serializer=message__pb2.Empty.SerializeToString,
            ),
            'send': grpc.unary_stream_rpc_method_handler(
                    servicer.send,
                    request_deserializer=message__pb2.Message.FromString,
                    response_serializer=message__pb2.Message.SerializeToString,
            ),
            'send_stream': grpc.stream_stream_rpc_method_handler(
                    servicer.send_stream,
                    request_deserializer=message__pb2.Message.FromString,
                    response_serializer=message__pb2.Message.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'grpc.MessageBoard', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MessageBoard(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/grpc.MessageBoard/ping',
            message__pb2.Empty.SerializeToString,
            message__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def send(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/grpc.MessageBoard/send',
            message__pb2.Message.SerializeToString,
            message__pb2.Message.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def send_stream(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/grpc.MessageBoard/send_stream',
            message__pb2.Message.SerializeToString,
            message__pb2.Message.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
