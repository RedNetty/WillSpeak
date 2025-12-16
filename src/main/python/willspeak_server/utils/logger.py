"""
Logger utility for WillSpeak server
"""
import logging
import sys
from pathlib import Path

def setup_logger():
    """Set up and configure the logger"""
    logger = logging.getLogger("willspeak")
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create file handler
    log_dir = Path(__file__).parent.parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "willspeak_server.log")
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def get_logger(name):
    """Get a logger for a specific module"""
    return logging.getLogger(f"willspeak.{name}")