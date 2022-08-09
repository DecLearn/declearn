# coding: utf-8

"""gRPC MessageBoard Service implementing messages passing.

The `Service` class implemented here is a grpc MessageBoard designed
to handle messages transmitted over gRPC using the models defined in
a protobuf file (and derived python tools auto-generated by the grpc
compiler).
"""

import asyncio
import json
from concurrent import futures
from typing import Any, Dict, List, Optional

import grpc  # type: ignore

from declearn2.communication.api import flags
from declearn2.communication.grpc.protobufs.message_pb2 import (
    CheckMessageRequest, Empty, Error, JoinRequest, JoinReply, Message
)
from declearn2.communication.grpc.protobufs.message_pb2_grpc import (
    MessageBoard, add_MessageBoardServicer_to_server
)
from declearn2.utils import json_unpack


class Service(MessageBoard):  # revise: inherit MessageBoardServicer?
    """A gRPC MessageBoard service to be used by GrpcServer instances."""

    def __init__(
            self,
            nb_clients: int,
            host: str = 'localhost',
            port: int = 0,
            credentials: Optional[grpc.ChannelCredentials] = None,
        ) -> None:
        """Instantiate the gRPC service."""
        self.nb_clients = nb_clients
        self.host = host
        self.port = port
        self.credentials = credentials
        # Set up a grpc Server based on the previous attributes.
        self.server = self._setup_server()
        # Set up clients and incoming / outgoing messages registries.
        self.registered_users = {}  # type: Dict[str, Dict[str, Any]]
        self.outgoing_messages = {}  # type: Dict[str, List[Dict[str, Any]]]
        self.incoming_messages = {}  # type: Dict[str, Dict[str, Any]]
        #self.accept_new_clients = False  # revise: garbage-collect

    def _setup_server(
            self,
        ) -> grpc.Server:
        """Set up and return a grpc Server to be used by this service."""
        server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        add_MessageBoardServicer_to_server(self, server)  # type: ignore
        address = f'{self.host}:{self.port}'
        if self.credentials:
            self.port = server.add_secure_port(address, self.credentials)
        else:
            self.port = server.add_insecure_port(address)
        return server

    async def start(
            self,
        ) -> None:
        """Start the grpc.Server wrapped by this service."""
        await self.server.start()

    async def stop(
            self,
        ) -> None:
        """Stop the grpc.Server wrapped by this service."""
        await self.server.stop(grace=None)

    def ping(
            self,
            request: Empty,
            context: grpc.ServicerContext,
        ) -> Empty:
        """Handle a ping request."""
        if context.peer() not in self.registered_users:
            return Error(message=flags.FLAG_REFUSE_CONNECTION)
        return Empty()

    def join(
            self,
            request: JoinRequest,
            context: grpc.ServicerContext,
        ) -> JoinReply:
        """Handle a join request."""
        # NOTE: comment this to be able to run tests with several
        #       client whereas they share this same value
        if context.peer() in self.registered_users:  # fixme: add flag
            #return JoinReply(message=flags.FLAG_ALREADY_REGISTERED)
            return JoinReply(message=flags.FLAG_REFUSE_CONNECTION)
        # If clients are still welcome, register the user.
        if len(self.registered_users) < self.nb_clients:
            # Deserialize the received information.
            info = {"name": request.name}
            info.update(json.loads(request.info, object_hook=json_unpack))
            # Store information about the user and set up a messages stack.
            self.registered_users[context.peer()] = info
            self.outgoing_messages[request.name] = []
            # Return a positive JoinReply.
            return JoinReply(message=flags.FLAG_WELCOME)
        # Otherwise, return a negative JoinReply.
        return JoinReply(message=flags.FLAG_REFUSE_CONNECTION)

    async def check_message(
            self,
            request: CheckMessageRequest,
            context: grpc.ServicerContext,
        ) -> Message:
        """Handle a Message-checking request from a client."""
        if context.peer() not in self.registered_users:
            return Error(message=flags.FLAG_REFUSE_CONNECTION)
        # Stay idle until at least one message is available.
        client_name = request.name
        while not self.outgoing_messages[client_name]:
            await asyncio.sleep(1)
        # Unstack the oldest available message and return it.
        message = self.outgoing_messages[client_name].pop(0)
        return Message(action=message['action'], params=message['params'])

    async def send_message(
            self,
            request: Message,
            context: grpc.ServicerContext,
        ) -> Empty:
        """Handle a Message-sending request from a client."""
        if context.peer() not in self.registered_users:
            return Error(message=flags.FLAG_REFUSE_CONNECTION)
        # Deserialize the message's content and make it available.
        message = json.loads(request.params, object_hook=json_unpack)
        self.incoming_messages[request.name] = message
        # Ping back the message sender.
        return Empty()
