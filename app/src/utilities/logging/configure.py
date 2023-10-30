import logging
from pathlib import Path
from typing import Optional

from .format import LOG_FMT

__all__ = [
    "configure_logger",
]


def configure_logger(
        logfile: Optional[Path] = None, level: int = logging.DEBUG,
        stream: bool = True
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
    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    root.setLevel(level)
