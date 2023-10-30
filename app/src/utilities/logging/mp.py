import logging
from logging.handlers import QueueHandler
import multiprocessing as mp
from pathlib import Path
from typing import Optional

from .format import LOG_FMT

__all__ = [
    "configure_worker_logger",
    "configure_listener_logger",
]


def configure_worker_logger(
        queue: Optional[mp.Queue] = None, level: int = logging.DEBUG
):
    root = logging.getLogger()

    if not root.hasHandlers() and queue is not None:
        h = logging.handlers.QueueHandler(queue)
        root.addHandler(h)

    root.setLevel(level)


def configure_listener_logger(
        logfile: Optional[Path] = None, level: int = logging.DEBUG
):
    root = logging.getLogger()
    formatter = logging.Formatter(LOG_FMT)

    # write to log file
    if logfile is not None:
        if not logfile.parent.exists():
            logfile.parent.mkdir(parents=True)
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # write to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    root.setLevel(level)
