"""Utility helpers for spawn detection."""

from __future__ import annotations

import multiprocessing as mp
import inspect


def in_spawned_main_toplevel() -> bool:
    """Return True when running at module top-level inside a spawned child process."""

    spawned = (mp.parent_process() is not None) or (
        mp.current_process().name != "MainProcess"
    )
    if not spawned:
        return False
    # Inspect the caller; only treat direct module execution as top-level
    stack = inspect.stack()
    if len(stack) < 3:
        return False
    caller = stack[2]
    # Only consider the *main* module at top-level as the spawned toplevel.
    if caller.function != "<module>":
        return False
    g = caller.frame.f_globals
    name = g.get("__name__")
    if name in ("__main__", "__mp_main__"):
        return True
    # When executed as the main module, __spec__ is typically None.
    return g.get("__spec__") is None


def is_spawned() -> bool:
    """Return True when running in a spawned child process."""

    parent = mp.parent_process()
    if parent is not None:
        return True
    # Fallback: detect a spawned child by process name, but only when start method is spawn
    # to avoid false positives for arbitrary subprocesses.
    try:
        if mp.get_start_method(allow_none=True) == "spawn":
            return mp.current_process().name != "MainProcess"
    except Exception:
        pass
    return False
