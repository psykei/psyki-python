import logging

__all__ = [
    "logger",
    "enable_logging",
    "LOG_DEBUG",
    "LOG_INFO",
    "LOG_WARNING",
    "LOG_ERROR",
    "LOG_CRITICAL",
    "LOG_FATAL",
]

logger = logging.getLogger("psyki")
logger.setLevel(logging.DEBUG)
logger.propagate = False

LOG_DEBUG = logging.DEBUG
LOG_INFO = logging.INFO
LOG_WARNING = logging.WARNING
LOG_ERROR = logging.ERROR
LOG_CRITICAL = logging.CRITICAL
LOG_FATAL = logging.FATAL


def enable_logging(level: int = LOG_INFO):
    """
    Enable logging.
    @param level: the logging level.
    """
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def disable_logging():
    """
    Disable logging.
    """
    logger.setLevel(logging.CRITICAL)
    logger.handlers = []
