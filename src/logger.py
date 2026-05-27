"""
logger.py
---------
Centralised logging setup.  Import ``get_logger`` in every module.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


_RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_logger(name: str = "melway", log_dir: str = "./logs") -> logging.Logger:
    """Return a module-level logger that writes to stdout and a per-run log file.

    All modules share the same run-stamped file (one file per process run),
    so a fresh file is created on every pipeline invocation.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"pipeline_{_RUN_STAMP}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
