import logging
from logging.handlers import QueueHandler
import multiprocessing as mp

__all__ = [
    "worker_logger_configurer",
    "listener_logger_configurer",
    "LOG_FMT"
]

LOG_FMT = '[%(asctime)s]::(PID %(process)d)::%(levelname)-2s::%(message)s'


def worker_logger_configurer(queue: mp.Queue, level=logging.DEBUG):
    root = logging.getLogger()

    if not root.hasHandlers():
        h = logging.handlers.QueueHandler(queue)
        root.addHandler(h)

    root.setLevel(level)


def listener_logger_configurer(logfile, level=logging.DEBUG):
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
