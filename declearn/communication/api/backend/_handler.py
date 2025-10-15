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

"""Protocol-agnostic server-side network messages handler."""

import asyncio
import logging
import math
from typing import Any, Dict, Optional, Set, Union

from declearn.communication.api.backend import flags
from declearn.communication.api.backend.actions import (
    Accept,
    ActionMessage,
    Drop,
    Join,
    LegacyMessageError,
    LegacyReject,
    Ping,
    Recv,
    Reject,
    Send,
    parse_action_from_string,
)
from declearn.version import VERSION


class MessagesHandler:
    """Minimal protocol-agnostic server-side messages handler."""

    def __init__(
        self,
        logger: logging.Logger,
        heartbeat: float = 1.0,
    ) -> None:
        # Assign parameters as attributes.
        self.logger = logger
        self.heartbeat = heartbeat
        # Set up containers for client identifiers and pending messages.
        self.registered_clients: Dict[Any, str] = {}
        self.outgoing_messages: Dict[str, str] = {}
        self.incoming_messages: Dict[str, str] = {}
        # Mark client-registration as unopened.
        self.registration_status = flags.REGISTRATION_UNSTARTED

    @property
    def client_names(self) -> Set[str]:
        """Names of the registered clients."""
        return set(self.registered_clients.values())

    async def purge(
        self,
    ) -> None:
        """Close opened connections and purge information about users.

        This resets the instance as though it was first initialized.
        User registration will be marked as unstarted.
        """
        self.registered_clients.clear()
        self.outgoing_messages.clear()
        self.incoming_messages.clear()
        self.registration_status = flags.REGISTRATION_UNSTARTED

    async def handle_message(
        self,
        string: str,
        context: Any,
    ) -> ActionMessage:
        """Handle an incoming message from a client.

        Parameters
        ----------
        string: str
            Received message, as a string that can be parsed back
            into an `ActionMessage` instance.
        context: hashable
            Communications-protocol-specific hashable object that
            may be used to uniquely identify (and thereof contact)
            the client that sent the message being handled.

        Returns
        -------
        message: ActionMessage
            Message to return to the sender, the specific type of
            which depends on the type of incoming request, errors
            encountered, etc.
        """
        # Parse the incoming message. If it is incorrect, reject it.
        try:
            message = parse_action_from_string(string)
        except (KeyError, TypeError, ValueError) as exc:
            self.logger.info(
                "Exception encountered while parsing received message: %s",
                repr(exc),
            )
            return Reject(flags.INVALID_MESSAGE)
        except LegacyMessageError as exc:
            self.logger.info(repr(exc))
            return LegacyReject()
        # Case: join request from a (new) client. Handle it.
        if isinstance(message, Join):
            return await self._handle_join_request(message, context)
        # Case: unregistered client. Reject message.
        if context not in self.registered_clients:
            return Reject(flags.REJECT_UNREGISTERED)
        # Case: registered client. Handle it.
        return await self._handle_registered_client_message(message, context)

    async def _handle_registered_client_message(
        self,
        message: ActionMessage,
        context: Any,
    ) -> ActionMessage:
        """Backend to handle a message from a registered client."""
        # Case: message-receiving request. Handle it.
        if isinstance(message, Recv):
            return await self._handle_recv_request(message, context)
        # Case: message-sending request. Handle it.
        if isinstance(message, Send):
            return await self._handle_send_request(message, context)
        # Case: drop message from a client. Handle it.
        if isinstance(message, Drop):
            return await self._handle_drop_request(message, context)
        # Case: ping request. Ping back.
        if isinstance(message, Ping):
            return Ping()
        # Case: unsupported message. Reject it.
        self.logger.error(
            "TypeError: received a message of unexpected type '%s'",
            type(message).__name__,
        )
        return Reject(flags.INVALID_MESSAGE)

    async def _handle_join_request(
        self,
        message: Join,
        context: Any,
    ) -> Union[Accept, Reject]:
        """Handle a join request."""
        # Case when client is already registered: warn but send OK.
        if context in self.registered_clients:
            self.logger.info(
                "Client %s is already registered.",
                self.registered_clients[context],
            )
            return Accept(flags.REGISTERED_ALREADY)
        # Case when registration is not opened: warn and reject.
        if self.registration_status != flags.REGISTRATION_OPEN:
            self.logger.info("Rejecting registration request.")
            return Reject(flag=self.registration_status)
        # Case when the client uses an incompatible declearn version.
        if (err := self._verify_version_compatibility(message)) is not None:
            return err
        # Case when registration is opened: register the client.
        self._register_client(message, context)
        return Accept(flag=flags.REGISTERED_WELCOME)

    def _verify_version_compatibility(
        self,
        message: Join,
    ) -> Optional[Reject]:
        """Return an 'Error' if a 'JoinRequest' is of incompatible version."""
        if message.version.split(".")[:2] == VERSION.split(".")[:2]:
            return None
        self.logger.info(
            "Received a registration request under name %s, that is "
            "invalid due to the client using DecLearn '%s'.",
            message.name,
            message.version,
        )
        return Reject(flags.REJECT_INCOMPATIBLE_VERSION)

    def _register_client(
        self,
        message: Join,
        context: Any,
    ) -> None:
        """Register a user based on their Join request and context object."""
        # Alias the user name if needed to avoid duplication issues.
        name = message.name
        used = self.client_names
        if name in used:
            idx = sum(other.rsplit(".", 1)[0] == name for other in used)
            name = f"{name}.{idx}"
        # Register the user, recording context and received data information.
        self.logger.info("Registering client '%s' for training.", name)
        self.registered_clients[context] = name

    async def _handle_send_request(
        self,
        message: Send,
        context: Any,
    ) -> Union[Ping, Reject]:
        """Handle a message-sending request (client-to-server)."""
        name = self.registered_clients[context]
        # Wait for any previous message from this client to be collected.
        while self.incoming_messages.get(name):
            await asyncio.sleep(self.heartbeat)
        # Record the received message, and return a ping-back response.
        self.incoming_messages[name] = message.content
        return Ping()

    async def _handle_recv_request(
        self,
        message: Recv,
        context: Any,
    ) -> Union[Send, Reject]:
        """Handle a message-receiving request."""
        # Set up the optional timeout mechanism.
        timeout = message.timeout
        countdown = (
            max(math.ceil(timeout / self.heartbeat), 1) if timeout else -1
        )
        # Wait for a message to be available or timeout to be reached.
        name = self.registered_clients[context]
        while (not self.outgoing_messages.get(name)) and countdown:
            await asyncio.sleep(self.heartbeat)
            countdown -= 1
        # Either send back the collected message, or a timeout error.
        try:
            content = self.outgoing_messages.pop(name)
        except KeyError:
            return Reject(flags.CHECK_MESSAGE_TIMEOUT)
        return Send(content)

    async def _handle_drop_request(
        self,
        message: Drop,
        context: Any,
    ) -> Ping:
        """Handle a drop request from a client."""
        name = self.registered_clients.pop(context)
        reason = (
            f"reason: '{message.reason}'" if message.reason else "no reason"
        )
        self.logger.info("Client %s has dropped with %s.", name, reason)
        return Ping()

    def post_message(
        self,
        message: str,
        client: str,
    ) -> None:
        """Post a message to be requested by a given client.

        Parameters
        ----------
        message: str
            Message string that is to be posted for the client to collect.
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
        if client in self.outgoing_messages:
            self.logger.warning(
                "Overwriting pending message uncollected by client '%s'.",
                client,
            )
        self.outgoing_messages[client] = message

    async def send_message(
        self,
        message: str,
        client: str,
        timeout: Optional[float] = None,
    ) -> None:
        """Post a message for a client and wait for it to be collected.

        Parameters
        ----------
        message: str
            Message string that is to be posted for the client to collect.
        client: str
            Name of the client to whom the message is addressed.
        timeout: float or None, default=None
            Optional maximum delay (in seconds) beyond which to stop
            waiting for collection and raise an asyncio.TimeoutError.

        Raises
        ------
        asyncio.TimeoutError
            If `timeout` is set and is reached while the message is
            yet to be collected by the client.

        Notes
        -----
        See the `post_message` method to synchronously post a message
        and move on without guarantees that it was collected.
        """
        # Post the message. Wait for it to have been collected.
        self.post_message(message, client)
        countdown = (
            max(math.ceil(timeout / self.heartbeat), 1) if timeout else -1
        )
        while self.outgoing_messages.get(client, False) and countdown:
            await asyncio.sleep(self.heartbeat)
            countdown -= 1
        # If the message is still there, raise a TimeoutError.
        if self.outgoing_messages.get(client):
            raise asyncio.TimeoutError(
                "Timeout reached before the sent message was collected."
            )

    def check_message(
        self,
        client: str,
    ) -> Optional[str]:
        """Check whether a message was received from a given client.

        Parameters
        ----------
        client: str
            Name of the client whose emitted message to check for.

        Returns
        -------
        message:
            Collected message that was sent by `client`, if any.
            In case no message is available, return None.

        Notes
        -----
        See the `recv_message` async method to wait for a message
        from the client to be available, collect and return it.
        """
        if client not in self.client_names:
            raise KeyError(f"Unregistered checked-for client '{client}'.")
        return self.incoming_messages.pop(client, None)

    async def recv_message(
        self,
        client: str,
        timeout: Optional[float] = None,
    ) -> str:
        """Wait for a message to be received from a given client.

        Parameters
        ----------
        client: str
            Name of the client whose emitted message to check for.
        timeout: float or None, default=None
            Optional maximum delay (in seconds) beyond which to stop
            waiting for a message and raise an asyncio.TimeoutError.

        Raises
        ------
        asyncio.TimeoutError
            If `timeout` is set and is reached while no message has
            been received from the client.

        Returns
        -------
        message:
            Collected message that was sent by `client`.

        Notes
        -----
        See the `check_message` method to synchronously check whether
        a message from the client is available and return it or None.
        """
        countdown = (
            max(math.ceil(timeout / self.heartbeat), 1) if timeout else -1
        )
        while countdown:
            message = self.check_message(client)
            if message is not None:
                return message
            await asyncio.sleep(self.heartbeat)
            countdown -= 1
        raise asyncio.TimeoutError(
            "Timeout reached before a message was received."
        )

    def open_clients_registration(
        self,
    ) -> None:
        """Make this servicer accept registration of new clients."""
        self.registration_status = flags.REGISTRATION_OPEN

    def close_clients_registration(
        self,
    ) -> None:
        """Make this servicer reject registration of new clients."""
        self.registration_status = flags.REGISTRATION_CLOSED

    async def wait_for_clients(
        self,
        min_clients: int = 1,
        max_clients: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Wait for clients to register for training, with given criteria.

        Parameters
        ----------
        min_clients: int, default=1
            Minimum number of clients required. Corrected to be >= 1.
            If `timeout` is None, used as the exact number of clients
            required - once reached, registration will be closed.
        max_clients: int or None, default=None
            Maximum number of clients authorized to register.
        timeout: float or None, default=None
            Optional maximum waiting time (in seconds) beyond which
            to close registration and either return or raise.

        Raises
        ------
        RuntimeError
            If the number of registered clients does not abide by the
            provided boundaries at the end of the process.
        """
        # Ensure any collected information is purged in case of failure
        # (due to raised errors or wrong number of registered clients).
        try:
            await self._wait_for_clients(min_clients, max_clients, timeout)
        except Exception as exc:  # re-raise; pylint: disable=broad-except
            await self.purge()
            raise exc

    async def _wait_for_clients(
        self,
        min_clients: int = 1,
        max_clients: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Backend of `wait_for_clients` method, without safeguards."""
        # Parse information on the required number of clients.
        min_clients = max(min_clients, 1)
        max_clients = -1 if max_clients is None else max_clients
        if max_clients < 0:
            max_clients = (
                min_clients if timeout is None else math.inf  # type: ignore
            )
        else:
            max_clients = max(min_clients, max_clients)
        # Wait for the required number of clients to have joined.
        self.open_clients_registration()
        countdown = (
            max(math.ceil(timeout / self.heartbeat), 1) if timeout else -1
        )
        while countdown and (len(self.registered_clients) < max_clients):
            await asyncio.sleep(self.heartbeat)
            countdown -= 1
        self.close_clients_registration()
        # Check whether all requirements have been checked.
        n_clients = len(self.registered_clients)
        if not min_clients <= n_clients <= max_clients:
            raise RuntimeError(
                f"The number of registered clients is {n_clients}, which "
                f"is out of the [{min_clients}, {max_clients}] range."
            )
