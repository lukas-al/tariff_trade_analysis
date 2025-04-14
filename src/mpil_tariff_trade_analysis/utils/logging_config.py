"""
Logging configuration for the MPIL Tariff Trade Analysis project.

This module sets up logging for the entire project, writing logs to both a file
in the top-level logs directory and to stdout (with color).
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Define log levels and their names for easier reference
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# ANSI escape codes for colors
COLORS = {
    logging.DEBUG: "\x1b[34m",  # Blue for DEBUG
    logging.INFO: "\x1b[32m",  # Green for INFO
    logging.WARNING: "\x1b[33m",  # Yellow for WARNING
    logging.ERROR: "\x1b[31m",  # Red for ERROR
    logging.CRITICAL: "\x1b[41m",  # Red background for CRITICAL
}
RESET = "\x1b[0m"  # Reset the color at the end of the message

# Default log level
DEFAULT_LOG_LEVEL = "DEBUG"

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Log directory
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure log directory exists
LOG_DIR.mkdir(exist_ok=True)

# Log file path
LOG_FILE = LOG_DIR / "mpil_tariff_trade_analysis.log"

# Flag to track if logging has been set up
_logging_initialized = False


class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter that adds color based on log level.
    Only applies color if the output stream is a TTY.
    """

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, *, defaults=None):
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.use_color = sys.stdout.isatty() # Check if output is a terminal

    def format(self, record):
        # Get the original formatted message
        log_msg = super().format(record)

        # Apply color if it's a TTY
        if self.use_color and record.levelno in COLORS:
            return f"{COLORS[record.levelno]}{log_msg}{RESET}"
        else:
            return log_msg


def setup_logging(log_level=None):
    """
    Set up logging configuration for the project.

    Args:
        log_level (str, optional): Log level to use. Defaults to environment variable or INFO.

    Returns:
        logging.Logger: Configured root logger
    """
    global _logging_initialized

    # If logging is already initialized, just return the root logger
    if _logging_initialized:
        return logging.getLogger()

    # Get log level from parameter, environment variable, or default
    if log_level is None:
        log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)

    # Convert string log level to logging constant
    numeric_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear any existing handlers to avoid duplicate logs
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Create formatters
    # File formatter remains standard
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    # Console formatter uses the new ColoredFormatter
    console_format = "%(asctime)s - %(levelname)s - %(message)s"
    console_formatter = ColoredFormatter(fmt=console_format)

    # Create file handler for logging to a file (with rotation)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,  # 10MB max size, keep 5 backups
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter) # Use standard formatter for file

    # Create console handler for logging to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter) # Use colored formatter for console

    # Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Log initial message
    root_logger.info(f"Logging initialized at level {log_level}")

    # Mark logging as initialized
    _logging_initialized = True

    return root_logger


def get_logger(name):
    """
    Get a logger for a specific module.
    If logging hasn't been set up yet, it will automatically set it up.

    Args:
        name (str): Name of the module (typically __name__)

    Returns:
        logging.Logger: Logger for the specified module
    """
    # Set up logging if it hasn't been initialized yet
    if not _logging_initialized:
        setup_logging()

    return logging.getLogger(name)
