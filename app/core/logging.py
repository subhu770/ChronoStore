"""
Logging configuration for ChronoStore.
Provides structured, standardized logging across all modules.
"""

import logging
import sys
from typing import Optional
from app.core.config import settings


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns the root application logger.
    Uses ISO-like timestamp formatting and clean stdout streaming.
    """
    level_name = (log_level or settings.LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger = logging.getLogger("chronostore")
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        logger.addHandler(handler)
    else:
        logger.handlers.clear()
        logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logging()

