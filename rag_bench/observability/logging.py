"""Structured logging configuration using structlog.

Provides JSON logging in production and colored console output in development.
Integrates with stdlib logging so all existing logger.info() calls across
the codebase automatically get structured formatting.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def setup_logging(
    log_level: str = "INFO",
    json_logs: bool | None = None,
) -> None:
    """Configure structlog with stdlib integration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_logs: Force JSON output. If None, auto-detect from RAG_ENV
                   (production = JSON, everything else = console).
    """
    if json_logs is None:
        json_logs = os.environ.get("RAG_ENV", "development") == "production"

    level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors for both structlog and stdlib
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to use structlog's formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers
    for name in ("uvicorn.access", "chromadb", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given name."""
    return structlog.get_logger(name)
