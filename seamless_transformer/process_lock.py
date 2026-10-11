"""Per-process serialization for transformation bodies."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
import threading
from typing import Callable, Iterator

from seamless.error_envelope import ExecutionCanceledError as TransformationCancelledError


NOTIFY_ENV = "SEAMLESS_PROCESS_LOCK_NOTIFY"
_ACQUIRE_INTERVAL = 0.05
_PROCESS_LOCK = threading.Lock()
_NOTIFY_GUARD = threading.Lock()
_NOTIFIED_PID: int | None = None
_CURRENT_HOLDER: ContextVar[_LockHolder | None] = ContextVar(
    "seamless_process_lock_holder", default=None
)


def _reset_notify_state_after_fork() -> None:
    global _NOTIFY_GUARD, _NOTIFIED_PID
    _NOTIFY_GUARD = threading.Lock()
    _NOTIFIED_PID = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_notify_state_after_fork)


class _LockHolder:
    """A shareable, one-way handle to a process-lock claim."""

    def __init__(self) -> None:
        self._held = True
        self._guard = threading.Lock()

    @property
    def held(self) -> bool:
        with self._guard:
            return self._held

    def drop(self) -> bool:
        """Release the lock once, even when called from another thread."""
        with self._guard:
            if not self._held:
                return False
            self._held = False
        _PROCESS_LOCK.release()
        return True


def current_holder() -> _LockHolder | None:
    """Return the holder associated with the current context, if any."""
    return _CURRENT_HOLDER.get()


@contextmanager
def scope(
    claim: bool, *, should_abort: Callable[[], bool] | None = None
) -> Iterator[None]:
    """Set the body's lock context and release any claim when it exits.

    A body that does not claim the lock masks any holder inherited from its
    caller. A claimed holder can be dropped by nested work and is never
    reacquired.
    """
    outer_token = _CURRENT_HOLDER.set(None)
    holder: _LockHolder | None = None
    holder_token = None
    try:
        if claim:
            while True:
                if should_abort is not None and should_abort():
                    raise TransformationCancelledError("Transformation was canceled")
                if _PROCESS_LOCK.acquire(timeout=_ACQUIRE_INTERVAL):
                    break
            if should_abort is not None and should_abort():
                _PROCESS_LOCK.release()
                raise TransformationCancelledError("Transformation was canceled")
            holder = _LockHolder()
            holder_token = _CURRENT_HOLDER.set(holder)
        yield
    finally:
        if holder is not None:
            holder.drop()
        if holder_token is not None:
            _CURRENT_HOLDER.reset(holder_token)
        _CURRENT_HOLDER.reset(outer_token)


def drop_for_nested_work() -> None:
    """Give up the current body's process lock before it submits work."""
    global _NOTIFIED_PID
    holder = current_holder()
    if holder is not None:
        holder.drop()
        return

    notify_path = os.environ.get(NOTIFY_ENV)
    if not notify_path:
        return

    pid = os.getpid()
    with _NOTIFY_GUARD:
        if _NOTIFIED_PID == pid:
            return
        # Mark before I/O: signaling is best-effort and a single attempt per
        # process is enough, even when the pipe has already disappeared.
        _NOTIFIED_PID = pid

    try:
        notify_fd = os.open(notify_path, os.O_WRONLY | os.O_NONBLOCK)
    except Exception:
        return
    try:
        os.write(notify_fd, b"\x00")
    except Exception:
        pass
    finally:
        try:
            os.close(notify_fd)
        except Exception:
            pass


__all__ = ["NOTIFY_ENV", "scope", "current_holder", "drop_for_nested_work"]
