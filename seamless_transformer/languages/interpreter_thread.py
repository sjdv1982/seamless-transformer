"""Run interpreter calls on a stable thread for thread-affine runtimes."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor


_executors: dict[tuple[int, str], ThreadPoolExecutor] = {}
_executors_lock = threading.Lock()
_thread_state = threading.local()


def _get_executor(key: tuple[int, str]) -> ThreadPoolExecutor:
    executor = _executors.get(key)
    if executor is not None:
        return executor
    with _executors_lock:
        executor = _executors.get(key)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"seamless-interpreter-{key[1]}",
            )
            _executors[key] = executor
    return executor


def run_in_interpreter_thread(name, func, /, *args, **kwargs):
    """Run ``func`` on one process-wide thread associated with ``name``.

    Calls made from the owner thread run inline. Set
    ``SEAMLESS_INTERPRETER_THREAD=0`` to run inline on the calling thread.
    """

    if os.environ.get("SEAMLESS_INTERPRETER_THREAD") == "0":
        return func(*args, **kwargs)

    key = (os.getpid(), str(name))
    if getattr(_thread_state, "key", None) == key:
        return func(*args, **kwargs)

    def invoke():
        previous_key = getattr(_thread_state, "key", None)
        _thread_state.key = key
        try:
            return func(*args, **kwargs)
        finally:
            if previous_key is None:
                try:
                    del _thread_state.key
                except AttributeError:
                    pass
            else:
                _thread_state.key = previous_key

    return _get_executor(key).submit(invoke).result()


__all__ = ["run_in_interpreter_thread"]
