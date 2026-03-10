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

"""Logging tools for declearn internal use."""

import logging
import os
import warnings
from typing import Optional

# TODO for 2.10: remove get_logger from list
__all__ = [
    "get_logger",
    "LOGGING_LEVEL_MAJOR",
    "config_logger",
    "config_server_loggers",
    "config_client_loggers",
]


# Add a logging level between INFO and WARNING.
LOGGING_LEVEL_MAJOR = (logging.WARNING + logging.INFO) // 2
"""Custom "MAJOR" severity level, between stdlib "INFO" and "WARNING"."""
logging.addLevelName(level=LOGGING_LEVEL_MAJOR, levelName="MAJOR")

DEFAULT_FORMAT = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


# TODO for 2.10: deprecated, remove function
def get_logger(
    name: str,
    level: int = logging.INFO,
    fpath: Optional[str] = None,
    s_fmt: Optional[str] = None,
) -> logging.Logger:
    """Create or access a logging.Logger instance with pre-set handlers.

    Deprecated since v2.8, will be removed in v2.10

    Parameters
    ----------
    name: str
        Name of the logger (used to create or retrieve it).
    level: int, default=logging.INFO
        Logging level below which messages are filtered out.
    fpath: str or None, default=None
        Optional path to a utf-8 text file to which to append
        logged messages (in addition to stream display).
    s_fmt: str or None, default=None
        Optional format string applied to the handlers.
        If None, use the default format set by declearn.

    Returns
    -------
    logger: logging.Logger
        Retrieved or created Logger, with a StreamHandler, opt.
        a FileHandler, and possibly more (if pre-existing).
    """
    warnings.warn(
        "get_logger() is deprecated and will be removed in 2.10",
        DeprecationWarning,
        stacklevel=2,
    )
    # Create or access the logger. Set its filtering level.
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Create or update an associated stream handler using the proper format.
    formatter = logging.Formatter(s_fmt or DEFAULT_FORMAT)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)
            break
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Optionally add a file handler, with similar formatting.
    if fpath:
        folder = os.path.dirname(os.path.abspath(fpath))
        os.makedirs(folder, exist_ok=True)
        handler = logging.FileHandler(fpath, mode="a", encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Return the logger instance.
    return logger


def config_logger(
    name: str,
    level: Optional[int] = None,
    fpath: Optional[str] = None,
    s_fmt: Optional[str] = None,
) -> None:
    """Util function to easily get the logger with the provided named and setup
    it with a basic configuration and pre-set handlers.

    Add a stream handler (logs on standard error output) if none exists for the
    provided logger name.

    Parameters
    ----------
    name: str
        Name of the logger to configure.
    level: int or None
        New logging level to apply.
    fpath: str or None
        Optional file to log messages to.
    s_fmt: str or None
        Optional format string for all handlers.
        If None, use the default format set by Declearn.
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    formatter = logging.Formatter(s_fmt or DEFAULT_FORMAT)

    # Update existing stream handlers.
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)

    # Add a stream handler if none exist.
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    # Add a file handler if requested and not already present.
    if fpath and not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == os.path.abspath(fpath)
        for h in logger.handlers
    ):
        os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
        fh = logging.FileHandler(fpath, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def config_server_loggers(
    level: Optional[int] = None,
    fpath: Optional[str] = None,
    s_fmt: Optional[str] = None,
):
    """Util function to easy apply the same simple configuration to all
    loggers related to the federated server.
    """
    config_logger("declearn.server", level=level, fpath=fpath, s_fmt=s_fmt)


def config_client_loggers(
    client_name: str,
    level: Optional[int] = None,
    fpath: Optional[str] = None,
    s_fmt: Optional[str] = None,
):
    """Util function to easy apply the same simple configuration to all
    loggers related to the client that match the provided client name.
    """
    config_logger(
        f"declearn.client-{client_name}", level=level, fpath=fpath, s_fmt=s_fmt
    )
