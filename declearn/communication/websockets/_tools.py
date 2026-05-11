# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

from websockets.legacy.protocol import WebSocketCommonProtocol

__all__ = [
    "StreamRefusedError",
    "receive_websockets_message",
    "send_websockets_message",
]

CHUNK_LENGTH = 2**20 - 16  # websocket max_size - overhead with a safety margin

FLAG_STREAM_START = b"STREAM_START"
FLAG_STREAM_CLOSE = b"STREAM_CLOSE"
FLAG_STREAM_ALLOW = b"STREAM_ALLOW"
FLAG_STREAM_BLOCK = b"STREAM_BLOCK"


class StreamRefusedError(Exception):
    """Custom Exception to signal cases when chunks-streaming was refused."""


async def receive_websockets_message(
    bin_msg: bytes,
    socket: WebSocketCommonProtocol,
    allow_chunks: bool = False,
) -> bytes:
    """Process a message received from an open socket.

    Parameters
    ----------
    bin_msg : bytes
        Initial message received through `socket`.
    socket : WebSocketCommonProtocol
        Open socket through which `message` was received
    allow_chunks : bool, default=False
        Whether to allow chunks-streaming, i.e. triggering a series
        of message-receiving calls to assemble a large message from
        a sequence of chunks (if `message` is a specific flag).

    Returns
    -------
    bin_msg: bytes
        The received message, which may be `message` or the result
        of a chunks-streaming operation.
    """
    if bin_msg == FLAG_STREAM_START:
        if not allow_chunks:
            await socket.send(FLAG_STREAM_BLOCK)
            raise StreamRefusedError(
                "Received a disallowed request to stream a chunked message."
            )
        await socket.send(FLAG_STREAM_ALLOW)
        chunks = []
        while True:
            buffer = await socket.recv()
            if buffer == FLAG_STREAM_CLOSE:
                break
            chunks.append(buffer)
        bin_msg = b"".join(chunks)
    return bin_msg


async def send_websockets_message(
    bin_msg: bytes,
    socket: WebSocketCommonProtocol,
) -> None:
    """Send a message through an open socket.

    Parameters
    ----------
    bin_msg : bytes
        Binary content to send.
    socket : WebSocketCommonProtocol
        Open socket through which `message` is to be sent.
    """
    if len(bin_msg) > CHUNK_LENGTH:
        # subtract overhead size with a safety margin
        await socket.send(FLAG_STREAM_START)
        if await socket.recv() != FLAG_STREAM_ALLOW:
            raise StreamRefusedError(
                "Message required chunking, but chunks-streaming was "
                "disallowed by the remote endpoint."
            )
        # Create a memoryview to chunk without allocating new byte copies.
        view = memoryview(bin_msg)
        for srt in range(0, len(bin_msg), CHUNK_LENGTH):
            end = srt + CHUNK_LENGTH
            await socket.send(view[srt:end])
        await socket.send(FLAG_STREAM_CLOSE)
    else:
        await socket.send(bin_msg)
