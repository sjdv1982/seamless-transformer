"""Per-process serialization for transformation bodies."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import threading
from typing import Callable, Iterator

from seamless.error_envelope import ExecutionCanceledError as TransformationCancelledError


NOTIFY_ENV = "SEAMLESS_PROCESS_LOCK_NOTIFY"
_ACQUIRE_INTERVAL = 0.05
_PROCESS_LOCK = threading.Lock()
_CURRENT_HOLDER: ContextVar[_LockHolder | None] = ContextVar(
    "seamless_process_lock_holder", default=None
)


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
    holder = current_holder()
    if holder is not None:
        holder.drop()


__all__ = ["NOTIFY_ENV", "scope", "current_holder", "drop_for_nested_work"]
