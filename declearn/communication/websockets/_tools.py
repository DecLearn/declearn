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

"""Shared backend utils for Websockets communication endpoints."""

import sys
from typing import Union

from websockets.legacy.protocol import WebSocketCommonProtocol

__all__ = [
    "StreamRefusedError",
    "receive_websockets_message",
    "send_websockets_message",
]


FLAG_STREAM_START = "STREAM_START"
FLAG_STREAM_CLOSE = "STREAM_CLOSE"
FLAG_STREAM_ALLOW = "STREAM_ALLOW"
FLAG_STREAM_BLOCK = "STREAM_BLOCK"


class StreamRefusedError(Exception):
    """Custom Exception to signal cases when chunks-streaming was refused."""


async def receive_websockets_message(
    message: Union[str, bytes],
    socket: WebSocketCommonProtocol,
    allow_chunks: bool = False,
) -> str:
    """Process a message received from an open socket.

    Parameters
    ----------
    message : Union[str, bytes]
        Initial message received through `socket`.
    socket : WebSocketCommonProtocol
        Open socket through which `message` was received
    allow_chunks : bool, default=False
        Whether to allow chunks-streaming, i.e. triggering a series
        of message-receiving calls to assemble a large message from
        a sequence of chunks (if `message` is a specific flag).

    Returns
    -------
    message: str
        The received message, which may be `message` or the result
        of a chunks-streaming operation.
    """
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    if message == FLAG_STREAM_START:
        if not allow_chunks:
            await socket.send(FLAG_STREAM_BLOCK)
            raise StreamRefusedError(
                "Received a disallowed request to stream a chunked message."
            )
        await socket.send(FLAG_STREAM_ALLOW)
        buffer = ""
        message = ""
        while buffer != FLAG_STREAM_CLOSE:
            message += buffer
            chunk = await socket.recv()
            buffer = (
                chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            )
    return message


async def send_websockets_message(
    message: str,
    socket: WebSocketCommonProtocol,
) -> None:
    """Send a message through an open socket.

    Parameters
    ----------
    message : str
        String content to send.
    socket : WebSocketCommonProtocol
        Open socket through which `message` is to be sent.
    """
    if socket.max_size and (sys.getsizeof(message) > socket.max_size):
        chunk_len = socket.max_size - sys.getsizeof("") - 1
        await socket.send(FLAG_STREAM_START)
        if await socket.recv() != FLAG_STREAM_ALLOW:
            raise StreamRefusedError(
                "Message required chunking, but chunks-streaming was "
                "disallowed by the remote endpoint."
            )
        for srt in range(0, len(message), chunk_len):
            end = srt + chunk_len
            await socket.send(message[srt:end])
        await socket.send(FLAG_STREAM_CLOSE)
    else:
        await socket.send(message)
