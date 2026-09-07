"""Port of ``src/utils/logging.{h,cc}`` (plog -> python logging)."""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "ptzcalib"

logger = logging.getLogger(_LOGGER_NAME)


def init_logging(verbose: bool = False, logfile: str | None = None) -> logging.Logger:
    """``InitLogging``: ``verbose=True`` maps to plog's debug severity."""
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname).1s %(message)s", "%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    if logfile:
        fhandler = logging.FileHandler(logfile, mode="w")
        fhandler.setFormatter(fmt)
        logger.addHandler(fhandler)

    return logger


# plog-style shorthands
def PLOGD(msg: str) -> None:
    logger.debug(msg)


def PLOGI(msg: str) -> None:
    logger.info(msg)


def PLOGW(msg: str) -> None:
    logger.warning(msg)


def PLOGE(msg: str) -> None:
    logger.error(msg)
