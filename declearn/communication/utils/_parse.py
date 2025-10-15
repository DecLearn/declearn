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

"""Utils to type-check received messages from the server or clients."""

from typing import Dict, Type, TypeVar

from declearn.communication.api import NetworkClient, NetworkServer
from declearn.messaging import Error, Message, SerializedMessage

__all__ = [
    "ErrorMessageException",
    "MessageTypeException",
    "verify_client_messages_validity",
    "verify_server_message_validity",
]


class ErrorMessageException(Exception):
    """Exception raised when an unexpected 'Error' message is received."""


class MessageTypeException(Exception):
    """Exception raised when a received 'Message' has wrong type."""


MessageT = TypeVar("MessageT", bound=Message)


async def verify_client_messages_validity(
    netwk: NetworkServer,
    received: Dict[str, SerializedMessage],
    expected: Type[MessageT],
) -> Dict[str, MessageT]:
    """Verify that received serialized messages match an expected type.

    - If all received messages matches expected type, deserialize them.
    - If any received message is an unexpected `Error` message, send an
      `Error` to non-error-send clients, then raise.
    - If any received message belongs to any other type, send an `Error`
      to each and every client, then raise.

    Parameters
    ----------
    netwk:
        `NetworkClient` endpoint, from which the processed message
        was received.
    received:
        Received `SerializedMessage` to type-check and deserialize.
    expected:
        Expected `Message` subtype. Any subclass will be considered
        as valid.

    Returns
    -------
    messages:
        Deserialized messages from `received`, with `expected` type,
        wrapped as a `{client_name: client_message}` dict.

    Raises
    ------
    ErrorMessageException
        If any `received` message wraps an unexpected `Error` message.
    MessageTypeException
        If any `received` wrapped message does not match `expected` type.
    """
    # Iterate over received messages to identify any unexpected 'Error' ones
    # or unexpected-type message.
    wrong_types = ""
    unexp_errors: Dict[str, str] = {}
    for client, srm in received.items():
        if issubclass(srm.message_cls, expected):
            pass
        elif issubclass(srm.message_cls, Error):
            unexp_errors[client] = srm.deserialize().message
        else:
            wrong_types += f"\n\t{client}: '{srm.message_cls}'"
    # In case of Error messages, send an Error to other clients and raise.
    if unexp_errors:
        await netwk.broadcast_message(
            Error("Some clients reported errors."),
            clients=set(received).difference(unexp_errors),
        )
        error = "".join(
            f"\n\t{key}:{val}" for key, val in unexp_errors.items()
        )
        raise ErrorMessageException(
            f"Expected '{expected.__name__}' messages, got the following "
            f"Error messages:{error}"
        )
    # In case of unproper messages, send an Error to all clients and raise.
    if wrong_types:
        error = (
            f"Expected '{expected.__name__}' messages, got the following "
            f"unproper message types:{wrong_types}"
        )
        await netwk.broadcast_message(Error(error), clients=set(received))
        raise MessageTypeException(error)
    # If everyting is fine, deserialized and return the received messages.
    return {cli: srm.deserialize() for cli, srm in received.items()}


async def verify_server_message_validity(
    netwk: NetworkClient,
    received: SerializedMessage,
    expected: Type[MessageT],
) -> MessageT:
    """Verify that a received serialized message matches expected type.

    - If the received message matches expected type, deserialize it.
    - If the recevied message is an unexpected `Error` message, raise.
    - If it belongs to any other type, send an `Error` to the server,
      then raise.

    Parameters
    ----------
    netwk:
        `NetworkClient` endpoint, from which the processed message
        was received.
    received:
        Received `SerializedMessage` to type-check and deserialize.
    expected:
        Expected `Message` subtype. Any subclass will be considered
        as valid.

    Returns
    -------
    message:
        Deserialized `Message` from `received`, with `expected` type.

    Raises
    ------
    ErrorMessageException
        If `received` wraps an unexpected `Error` message.
    MessageTypeException
        If `received` wrapped message does not match `expected` type.
    """
    # If a proper message is received, deserialize and return it.
    if issubclass(received.message_cls, expected):
        return received.deserialize()
    # When an Error is received, merely raise using its content.
    error = f"Expected a '{expected}' message"
    if issubclass(received.message_cls, Error):
        msg = received.deserialize()
        error = f"{error}, received an Error message: '{msg.message}'."
        raise ErrorMessageException(error)
    # Otherwise, send an Error to the server, then raise.
    error = f"{error}, got a '{received.message_cls}'."
    await netwk.send_message(Error(error))
    raise MessageTypeException(error)
