# coding: utf-8

"""Logging tools for declearn internal use."""

import logging
import os
from typing import Optional


DEFAULT_FORMAT = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def get_logger(
        name: str,
        level: int = logging.INFO,
        fpath: Optional[str] = None,
        s_fmt: Optional[str] = None,
    ) -> logging.Logger:
    """Create or access a logging.Logger instance with pre-set handlers.

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
