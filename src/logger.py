import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger.
    Outputs to stdout with a standard production format:
    [TIMESTAMP] - [MODULE] - [LEVEL] - [MESSAGE]
    """
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
