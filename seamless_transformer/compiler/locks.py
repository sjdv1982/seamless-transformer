"""Small file-based directory lock."""

from __future__ import annotations

from contextlib import contextmanager
import os
import time


@contextmanager
def directory_lock(directory: str, name: str = ".lock", poll: float = 0.05):
    os.makedirs(directory, exist_ok=True)
    lockfile = os.path.join(directory, name)
    fd = None
    while fd is None:
        try:
            fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            time.sleep(poll)
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.unlink(lockfile)
        except FileNotFoundError:
            pass


__all__ = ["directory_lock"]
