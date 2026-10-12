"""Cancellation cleanup callbacks for active bash transformations."""

from __future__ import annotations

import threading
from collections.abc import Callable

_lock = threading.RLock()
_cleanups: dict[str | None, dict[object, Callable[[], None]]] = {}


def _checksum_key(tf_checksum) -> str | None:
    if tf_checksum is None:
        return None
    value = getattr(tf_checksum, "hex", None)
    if callable(value):
        value = value()
    return value if isinstance(value, str) else str(tf_checksum)


def register_cleanup(tf_checksum, cleanup: Callable[[], None]) -> Callable[[], None]:
    """Register one launch cleanup and return an idempotent unregister callback.

    A checksum may have several live launches, so entries are kept separately.
    Running a cleanup does not unregister it: a racing launch can need the same
    idempotent cleanup again before its owning ``execute_bash`` reaches ``finally``.
    """

    key = _checksum_key(tf_checksum)
    token = object()
    with _lock:
        _cleanups.setdefault(key, {})[token] = cleanup

    def unregister() -> None:
        with _lock:
            callbacks = _cleanups.get(key)
            if callbacks is None:
                return
            callbacks.pop(token, None)
            if not callbacks:
                _cleanups.pop(key, None)

    return unregister


def _run(callbacks: list[Callable[[], None]]) -> int:
    ran = 0
    for cleanup in callbacks:
        try:
            cleanup()
        except Exception:
            # Cancellation cleanup is best-effort; a bad callback must not keep
            # the remaining process groups or containers alive.
            pass
        ran += 1
    return ran


def run_cleanups(tf_checksum) -> int:
    """Run every registered cleanup for one transformation checksum."""

    key = _checksum_key(tf_checksum)
    with _lock:
        callbacks = list(_cleanups.get(key, {}).values())
    return _run(callbacks)


def run_all_cleanups() -> int:
    """Run all launch cleanups in this process, for its SIGTERM handler."""

    with _lock:
        callbacks = [
            cleanup
            for group in _cleanups.values()
            for cleanup in group.values()
        ]
    return _run(callbacks)


__all__ = ["register_cleanup", "run_all_cleanups", "run_cleanups"]
