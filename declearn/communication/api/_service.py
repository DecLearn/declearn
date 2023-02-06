# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""Protocol-agnostic server-side network messages handler."""

import asyncio
import logging
from typing import Any, Dict, Optional, Set, Union


from declearn.communication.messaging import (
    Empty,
    Error,
    EvaluationReply,
    GenericMessage,
    GetMessageRequest,
    JoinReply,
    JoinRequest,
    Message,
    TrainReply,
    flags,
    parse_message_from_string,
)


__all__ = [
    "MessagesHandler",
]


class MessagesHandler:
    """Protocol-agnostic server-side network messages handler.

    This class implements generic mechanisms to:
    * manage an allow-list of registered clients
    * parse incoming messages from string into Message instances
    * handle messages based on their Message subclass, to either:
      - process the registration request from a client
      - make a message sent by the client available to the server
      - return a message posted by the server to an asking client
    * enable a managing server to:
      - post messages for clients to collect
      - collect messages sent by clients
    """

    def __init__(
        self,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self.registered_clients = {}  # type: Dict[Any, str]
        self.data_info = {}  # type: Dict[str, Dict[str, Any]]
        self.out_messages = {}  # type: Dict[str, Message]
        self.inc_messages = {}  # type: Dict[str, Message]
        # Declare attributes to be managed by the server class.
        self.registration_status = flags.REGISTRATION_UNSTARTED

    @property
    def client_names(self) -> Set[str]:
        """Names of the registered clients."""
        return set(self.registered_clients.values())

    async def purge(
        self,
    ) -> None:
        """Close opened connections and purge information about users."""
        self.registered_clients = {}
        self.data_info = {}
        self.out_messages = {}
        self.inc_messages = {}

    async def handle_message(
        self,
        string: str,
        context: Any,
    ) -> Message:
        """Handle an incoming message from a client.

        Parameters
        ----------
        string: str
            Received message, as a string that can be parsed back
            into a `declearn.communication.api.Message` instance.
        context: hashable
            Communications-protocol-specific hashable object that
            may be used to uniquely identify (and thereof contact)
            the client that sent the message being handled.

        Returns
        -------
        message: Message
            Message to return to the sender, the specific type of
            which depends on the type of incoming request, errors
            encountered, etc.
        """
        # case-switch function; pylint: disable=too-many-return-statements
        # Parse the incoming message. If it is incorrect, send back an error.
        try:
            message = parse_message_from_string(string)
        except (KeyError, TypeError) as exc:
            self.logger.info(
                "%s encountered while parsing received message: %s",
                type(exc).__name__,
                exc,
            )
            return Error(flags.INVALID_MESSAGE)
        # Return a message-type-based response.
        # Case: join request. Otherwise, reject messages from unknown clients.
        if isinstance(message, JoinRequest):
            return await self._handle_join_request(message, context)
        if context not in self.registered_clients:
            return Error(flags.REJECT_UNREGISTERED)
        # Case: ping request.
        if isinstance(message, Empty):
            return Empty()
        # Case: request from client to collect a posted message.
        if isinstance(message, GetMessageRequest):
            return await self._handle_recv_request(message, context)
        # Case: expected type of message being sent to server.
        acceptable = (Error, EvaluationReply, GenericMessage, TrainReply)
        if isinstance(message, acceptable):
            return await self._handle_send_request(message, context)
        # Otherwise, send back an error regarding incorrect message type.
        self.logger.error(
            "TypeError: received a message of unexpected type '%s'",
            type(message).__name__,
        )
        return Error(flags.INVALID_MESSAGE)

    async def _handle_join_request(
        self,
        message: JoinRequest,
        context: Any,
    ) -> JoinReply:
        """Handle a join request."""
        # Case when client is already registered: warn but send OK.
        if context in self.registered_clients:
            self.logger.info(
                "Client %s is already registered.",
                self.registered_clients[context],
            )
            reply = JoinReply(accept=True, flag=flags.REGISTERED_ALREADY)
        # Case when registration is not opened: warn and reject.
        elif self.registration_status != flags.REGISTRATION_OPEN:
            self.logger.info("Rejecting registration request.")
            reply = JoinReply(accept=False, flag=self.registration_status)
        # Case when registration is opened: register the client.
        else:
            self._register_client(message, context)
            reply = JoinReply(accept=True, flag=flags.REGISTERED_WELCOME)
        # Return the selected reply.
        return reply

    def _register_client(
        self,
        message: JoinRequest,
        context: Any,
    ) -> None:
        """Register a user based on their JoinRequest and context object."""
        # Alias the user name if needed to avoid duplication issues.
        name = message.name
        used = self.client_names
        if name in used:
            idx = sum(other.rsplit(".", 1)[0] == name for other in used)
            name = f"{name}.{idx}"
        # Register the user, recording context and received data information.
        self.logger.info("Registering client '%s' for training.", name)
        self.registered_clients[context] = name
        self.data_info[name] = message.data_info

    async def _handle_send_request(
        self,
        message: Message,
        context: Any,
    ) -> Union[Empty, Error]:
        """Handle a message-sending request (client-to-server)."""
        name = self.registered_clients[context]
        # Wait for any previous message from this client to be collected.
        while self.inc_messages.get(name):  # revise: remove? add timeout?
            await asyncio.sleep(1)
        # Record the received message, and return a ping-back response.
        self.inc_messages[name] = message
        return Empty()

    async def _handle_recv_request(
        self,
        message: GetMessageRequest,
        context: Any,
    ) -> Message:
        """Handle a message-receiving request."""
        # Set up the optional timeout mechanism.
        countdown = -1 if (message.timeout is None) else message.timeout
        # Wait for a message to be available or timeout to be reached.
        name = self.registered_clients[context]
        while (not self.out_messages.get(name)) and countdown:
            await asyncio.sleep(1)
            countdown -= 1
        # Either send back the collected message, or a timeout error.
        try:
            reply = self.out_messages.pop(name)  # type: Message
        except KeyError:
            reply = Error(flags.CHECK_MESSAGE_TIMEOUT)
        return reply

    def post_message(self, message: Message, client: str) -> None:
        """Post a message to be requested by a given client.

        Parameters
        ----------
        message: Message
            Message that is to be posted for the client to collect.
        client: str
            Name of the client to whom the message is addressed.

        Notes
        -----
        This method merely makes the message available for the client
        to request, without any guarantee that it is received.
        See the `send_message` async method to wait for the posted
        message to have been requested by and thus sent to the client.
        """
        if client not in self.client_names:
            raise KeyError(f"Unkown destinatory client '{client}'.")
        if client in self.out_messages:
            self.logger.warning(
                "Overwriting pending message uncollected by client '%s'.",
                client,
            )
        self.out_messages[client] = message

    async def send_message(
        self,
        message: Message,
        client: str,
        heartbeat: int = 1,
        timeout: Optional[int] = None,
    ) -> None:
        """Post a message for a client and wait for it to be collected.

        Parameters
        ----------
        message: Message
            Message that is to be posted for the client to collect.
        client: str
            Name of the client to whom the message is addressed.
        heartbeat: int, default=1
            Delay (in seconds) between verifications that the message
            has been collected by the client.
        timeout: int or None, default=None
            Optional maximum delay (in seconds) beyond which to stop
            waiting for collection and raise an asyncio.TimeoutError.

        Raises
        ------
        asyncio.TimeoutError:
            If `timeout` is set and is reached while the message is
            yet to be collected by the client.

        Notes
        -----
        See the `post_message` method to synchronously post a message
        and move on without guarantees that it was collected.
        """
        # Post the message. Wait for it to have been collected.
        self.post_message(message, client)
        countdown = -1
        if timeout is not None:
            countdown = (timeout // heartbeat) + bool(timeout % heartbeat)
        while self.out_messages.get(client, False) and countdown:
            await asyncio.sleep(heartbeat)
            countdown -= 1
        # If the message is still there, raise a TimeoutError.
        if self.out_messages.get(client):
            raise asyncio.TimeoutError(
                "Timeout reached before the sent message was collected."
            )

    def check_message(
        self,
        client: str,
    ) -> Optional[Message]:
        """Check whether a message was received from a given client.

        Parameters
        ----------
        client: str
            Name of the client whose emitted message to check for.

        Returns
        -------
        message: Message or None
            Collected message that was sent by `client`, if any.
            In case no message is available, return None.

        Notes
        -----
        See the `recv_message` async method to wait for a message
        from the client to be available, collect and return it.
        """
        if client not in self.client_names:
            raise KeyError(f"Unregistered checked-for client '{client}'.")
        return self.inc_messages.pop(client, None)

    async def recv_message(
        self,
        client: str,
        heartbeat: int = 1,
        timeout: Optional[int] = None,
    ) -> Message:
        """Wait for a message to be received from a given client.

        Parameters
        ----------
        client: str
            Name of the client whose emitted message to check for.
        heartbeat: int, default=1
            Delay (in seconds) between verifications that a message
            has been received from the client.
        timeout: int or None, default=None
            Optional maximum delay (in seconds) beyond which to stop
            waiting for a message and raise an asyncio.TimeoutError.

        Raises
        ------
        asyncio.TimeoutError:
            If `timeout` is set and is reached while no message has
            been received from the client.

        Returns
        -------
        message: Message
            Collected message that was sent by `client`.

        Notes
        -----
        See the `check_message` method to synchronously check whether
        a message from the client is available and return it or None.
        """
        countdown = -1
        if timeout is not None:
            countdown = (timeout // heartbeat) + bool(timeout % heartbeat)
        while countdown:
            message = self.check_message(client)
            if message is not None:
                return message
            await asyncio.sleep(heartbeat)
            countdown -= 1
        raise asyncio.TimeoutError(
            "Timeout reached before a message was received."
        )

    async def wait_for_clients(
        self,
        min_clients: int = 1,
        max_clients: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Wait for clients to register for training, with given criteria.

        Parameters
        ----------
        min_clients: int, default=1
            Minimum number of clients required. Corrected to be >= 1.
            If `timeout` is None, used as the exact number of clients
            required - once reached, registration will be closed.
        max_clients: int or None, default=None
            Maximum number of clients authorized to register.
        timeout: int or None, default=None
            Optional maximum waiting time (in seconds) beyond which
            to close registration and either return or raise.

        Raises
        ------
        RuntimeError:
            If the number of registered clients does not abide by the
            provided boundaries at the end of the process.

        Returns
        -------
        client_info: dict[str, dict[str, any]]
            A dictionary where the keys are the participants
            and the values are their information.
        """
        # Ensure any collected information is purged in case of failure
        # (due to raised errors or wrong number of registered clients).
        try:
            await self._wait_for_clients(min_clients, max_clients, timeout)
        except Exception as exc:  # re-raise; pylint: disable=broad-except
            self.registration_status = flags.REGISTRATION_UNSTARTED
            await self.purge()
            raise exc
        return self.data_info.copy()

    async def _wait_for_clients(
        self,
        min_clients: int = 1,
        max_clients: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Backend of `wait_for_clients` method, without safeguards."""
        # Parse information on the required number of clients.
        min_clients = max(min_clients, 1)
        if max_clients is None:
            if timeout is None:
                max_clients = min_clients
            else:
                max_clients = float("inf")  # type: ignore
        else:
            max_clients = max(min_clients, max_clients)
        # Wait for the required number of clients to have joined.
        self.registration_status = flags.REGISTRATION_OPEN
        countdown = timeout or -1
        while countdown:
            n_clients = len(self.registered_clients)
            if n_clients >= max_clients:  # type: ignore
                break
            await asyncio.sleep(1)
            countdown -= 1
        self.registration_status = flags.REGISTRATION_CLOSED
        # Check whether all requirements have been checked.
        n_clients = len(self.registered_clients)
        if not min_clients <= n_clients <= max_clients:  # type: ignore
            raise RuntimeError(
                f"The number of registered clients is {n_clients}, which "
                f"is out of the [{min_clients}, {max_clients}] range."
            )
