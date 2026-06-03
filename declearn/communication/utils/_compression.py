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

"""Shared helpers to resolve the declearn-level ``compression`` parameter
into transport-specific compression values for WebSockets and gRPC."""

from typing import Any, Optional

__all__ = [
    "VALID_COMPRESSION_VALUES",
    "resolve_grpc_compression",
    "resolve_ws_compression",
]


VALID_COMPRESSION_VALUES = (None, "none", "deflate")


def _validate(value: Any) -> Optional[str]:
    """Validate the declearn-level compression value.

    Returns the canonical form: None or "deflate". Raises ValueError on
    any other input.
    """
    if value is None or value == "none":
        return None
    if value == "deflate":
        return "deflate"
    raise ValueError(
        f"Invalid compression value: {value!r}. "
        f"Valid options are: {VALID_COMPRESSION_VALUES}."
    )


def resolve_ws_compression(value: Any) -> Optional[str]:
    """Translate a declearn compression value into a websockets-library value.

    Parameters
    ----------
    value:
        One of None, "none", "deflate".

    Returns
    -------
    None for no compression, or "deflate" for permessage-deflate.

    Raises
    ------
    ValueError
        If `value` is not one of the supported options.
    """
    return _validate(value)


def resolve_grpc_compression(value: Any) -> "Any":
    """Translate a declearn compression value into a `grpc.Compression` value.

    Parameters
    ----------
    value:
        One of None, "none", "deflate".

    Returns
    -------
    ``grpc.Compression.NoCompression`` when no compression is requested,
    or ``grpc.Compression.Deflate`` when ``"deflate"`` is requested.

    Raises
    ------
    ValueError
        If `value` is not one of the supported options.
    """
    canonical = _validate(value)
    # Lazy-import grpc: this module is also imported by the websockets
    # transport, which must work without grpc installed.
    import grpc  # type: ignore[import-untyped]  # noqa: PLC0415

    if canonical is None:
        return grpc.Compression.NoCompression
    return grpc.Compression.Deflate
