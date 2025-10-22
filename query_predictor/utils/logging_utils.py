"""Structured logging utilities for the query predictor service."""

import logging
import sys
from pythonjsonlogger.json import JsonFormatter
from typing import Dict, Any


def setup_logging(logging_config: Dict[str, Any]) -> None:
    """
    Set up structured JSON logging to stdout.

    Args:
        logging_config: Dictionary with 'level', 'format', 'output' keys
    """
    log_level = logging_config.get('level', 'INFO').upper()
    log_format = logging_config.get('format', 'json')
    log_output = logging_config.get('output', 'stdout')

    # Get numeric log level
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    if log_output == 'stdout':
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(numeric_level)

    # Create formatter
    if log_format == 'json':
        formatter = JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Log startup message
    logging.info(
        "Logging configured",
        extra={
            'log_level': log_level,
            'log_format': log_format,
            'log_output': log_output
        }
    )
