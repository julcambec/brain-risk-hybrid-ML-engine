"""Structured logging setup for the brainrisk package."""

from __future__ import annotations

import json
import logging as _logging
from typing import Any


class _JsonFormatter(_logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: _logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logger(
    name: str = "brainrisk",
    level: str = "INFO",
    json_format: bool = False,
) -> _logging.Logger:
    """Configure and return a logger with a console handler.

    If a handler is already attached to the named logger, the existing logger
    is returned without modification to avoid duplicate output.

    Parameters
    ----------
    name : str
        Logger name (typically ``"brainrisk"`` or a dotted sub-name).
    level : str
        Logging level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).
    json_format : bool
        If ``True``, emit structured JSON log lines instead of plain text.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = _logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(_logging, level.upper(), _logging.INFO))

    handler = _logging.StreamHandler()
    handler.setLevel(logger.level)

    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            _logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )

    logger.addHandler(handler)
    return logger
