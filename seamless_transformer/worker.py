"""Spawn and coordinate Seamless transformation workers.

This module builds on :mod:`seamless.transformer.process` to manage a pool of
worker processes. Workers execute transformation requests sent from the parent
process and forward buffer/checksum interactions back to the parent using
shared memory.

TODO: implement shared memory refcount
Would replace the current prefetch+write shared mem + worker download+confirm+cleanup cycle,
  which is transport-oriented, not resource-oriented.
Current mechanism for worker upload untouched.
Would be a seamless-base (buffer cache) feature.
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
import concurrent.futures as _cf
import traceback
import uuid
import sys
import logging
import multiprocessing as mp
import string
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker as _cf_worker
import weakref
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Dict, Optional

import seamless.caching.buffer_cache as _buffer_cache
import seamless.buffer_class as _buffer_class
from seamless import Buffer, CacheMissError, Checksum, set_is_worker, ensure_open
from seamless.caching.buffer_cache import get_buffer_cache

from .process import ChildChannel, ProcessManager, ConnectionClosed
from .process.manager import ProcessError
from .run import run_transformation_dict

_DASK_AVAILABLE_ENV = "SEAMLESS_DASK_AVAILABLE"
_ALLOW_REMOTE_CLIENTS_ENV = "SEAMLESS_ALLOW_REMOTE_CLIENTS_IN_WORKER"
_has_spawned = False
_dask_available = os.environ.get(_DASK_AVAILABLE_ENV) == "1"

_worker_manager: "_WorkerManager | None" = None
_child_channel: Optional[ChildChannel] = None
_child_loop: Optional[asyncio.AbstractEventLoop] = None
_primitives_patched = False
_dummy_buffer_cache = None
_transformer_registry: Dict[str, Any] = {}
_quiet = False
_DEBUG_SHUTDOWN = bool(os.environ.get("SEAMLESS_DEBUG_SHUTDOWN"))
_DELEGATION_REFUSED = "_DELEGATION_REFUSED"
_LOCAL_BUFFERS: Dict[str, bytes] = {}


# Throttle how many concurrent tasks a single worker can handle.
# Not to be changed dynamically
TRANSFORMATION_THROTTLE = int(
    os.environ.get("SEAMLESS_WORKER_TRANSFORMATION_THROTTLE", 3)
)


def _bump_nofile_limit(min_soft: int = 4096) -> None:
    """Try to lift the soft RLIMIT_NOFILE to reduce FD exhaustion."""

    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = max(soft, min_soft)
        if hard not in (-1, resource.RLIM_INFINITY):
            target = min(target, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        # Best-effort; ignore on platforms without resource or on failure.
        pass


_bump_nofile_limit()


def _memory_provider(_key: str) -> None:
    # Shared memory registry is unused for the worker pool.
    return None


@dataclass
class _Pointer:
    key: str
    shm: SharedMemory
    size: int
    metadata: Dict[str, Any]


def _close_shm(shm: SharedMemory) -> None:
    """Best-effort close wrapper that tolerates already-closed handles."""

    try:
        shm.close()
    except Exception:
        pass


def _require_child_channel() -> tuple[ChildChannel, asyncio.AbstractEventLoop]:
    if _child_channel is None or _child_loop is None:
        raise RuntimeError("Worker channel is not initialized")
    return _child_channel, _child_loop


def _is_checksum_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in string.hexdigits for ch in value)
    )


def _driver_flag_from_tf_dunder(tf_dunder: Any) -> bool:
    if not isinstance(tf_dunder, dict):
        return False
    meta = tf_dunder.get("__meta__")
    return isinstance(meta, dict) and bool(meta.get("driver"))


def _dependency_checksums_from_tf_dunder(tf_dunder: Any) -> Dict[str, str]:
    if not isinstance(tf_dunder, dict):
        return {}
    deps = tf_dunder.get("__deps__")
    if not isinstance(deps, dict):
        return {}
    parsed: Dict[str, str] = {}
    for key, value in deps.items():
        if isinstance(value, Checksum):
            value = value.hex()
        if isinstance(key, str) and isinstance(value, str) and value:
            parsed[key] = value
    return parsed


_DELEGATE_POLL_INTERVAL = float(
    os.environ.get("SEAMLESS_DELEGATE_POLL_INTERVAL", "0.2")
)
_DELEGATE_POLL_BATCH = int(os.environ.get("SEAMLESS_DELEGATE_POLL_BATCH", "200"))
_DELEGATE_STALE_TTL = float(os.environ.get("SEAMLESS_DELEGATE_STALE_TTL", "60"))
_DELEGATE_CLEANUP_INTERVAL = float(
    os.environ.get("SEAMLESS_DELEGATE_CLEANUP_INTERVAL", "10")
)
_DELEGATE_COMPLETED_TTL = float(
    os.environ.get("SEAMLESS_DELEGATE_COMPLETED_TTL", "300")
)
_DELEGATE_OWNER_CANCEL_TTL = float(
    os.environ.get("SEAMLESS_DELEGATE_OWNER_CANCEL_TTL", "60")
)
_DEFAULT_DASK_BASE_PRIORITY = 10
_DELEGATE_DEBUG_CLEANUP = bool(os.environ.get("SEAMLESS_DEBUG_DELEGATE_CLEANUP"))
_DELEGATE_PROXY_LOCK = threading.Lock()
_DELEGATE_PROXY_FUTURES: Dict[str, _cf.Future] = {}
_DELEGATE_POLL_TASK: asyncio.Task | None = None
_CURRENT_OWNER_DASK_KEY = threading.local()
_CURRENT_OWNER_DASK_PRIORITY = threading.local()


def _normalize_owner_dask_key(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith(("base-", "base_")):
        return value
    return None


def _normalize_owner_dask_priority(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _get_current_owner_dask_key() -> str | None:
    return getattr(_CURRENT_OWNER_DASK_KEY, "dask_key", None)


def _set_current_owner_dask_key(value: str | None) -> str | None:
    previous = getattr(_CURRENT_OWNER_DASK_KEY, "dask_key", None)
    if value is None:
        if hasattr(_CURRENT_OWNER_DASK_KEY, "dask_key"):
            delattr(_CURRENT_OWNER_DASK_KEY, "dask_key")
    else:
        _CURRENT_OWNER_DASK_KEY.dask_key = value
    return previous


def _get_current_owner_dask_priority() -> int | None:
    return getattr(_CURRENT_OWNER_DASK_PRIORITY, "priority", None)


def _set_current_owner_dask_priority(value: int | None) -> int | None:
    previous = getattr(_CURRENT_OWNER_DASK_PRIORITY, "priority", None)
    if value is None:
        if hasattr(_CURRENT_OWNER_DASK_PRIORITY, "priority"):
            delattr(_CURRENT_OWNER_DASK_PRIORITY, "priority")
    else:
        _CURRENT_OWNER_DASK_PRIORITY.priority = value
    return previous


def _scheduler_task_states(
    dask_scheduler, keys: list[str]
) -> Dict[str, tuple[str | None, bool | None]]:
    states: Dict[str, tuple[str | None, bool | None]] = {}
    for key in keys:
        ts = dask_scheduler.tasks.get(key)
        if ts is None:
            states[key] = (None, None)
            continue
        try:
            state = getattr(ts, "state", None)
        except Exception:
            state = None
        try:
            who_wants = bool(getattr(ts, "who_wants", None))
        except Exception:
            who_wants = None
        states[key] = (state, who_wants)
    return states


def _fetch_scheduler_task_states(
    raw_client: Any, keys: list[str]
) -> Dict[str, tuple[str | None, bool | None]]:
    return raw_client.run_on_scheduler(_scheduler_task_states, keys=keys)


def _owner_task_cancelled(state: str | None, who_wants: bool | None) -> bool:
    if state is None:
        return True
    if state in ("released", "forgotten", "erred", "cancelled"):
        return True
    if who_wants is False:
        return True
    return False


def _register_delegate_token(token: str) -> _cf.Future:
    fut: _cf.Future = _cf.Future()
    with _DELEGATE_PROXY_LOCK:
        _DELEGATE_PROXY_FUTURES[token] = fut
    _ensure_delegate_poller()
    return fut


def _ensure_delegate_poller() -> None:
    if _child_loop is None or _child_loop.is_closed():
        return

    def _start() -> None:
        global _DELEGATE_POLL_TASK
        if _DELEGATE_POLL_TASK is None or _DELEGATE_POLL_TASK.done():
            _DELEGATE_POLL_TASK = asyncio.create_task(_delegate_poll_loop())

    if asyncio.get_running_loop() is _child_loop:
        _start()
    else:
        _child_loop.call_soon_threadsafe(_start)


async def _delegate_poll_loop() -> None:
    while True:
        await asyncio.sleep(_DELEGATE_POLL_INTERVAL)
        with _DELEGATE_PROXY_LOCK:
            tokens = list(_DELEGATE_PROXY_FUTURES.keys())
            cancelled = [
                token
                for token, fut in _DELEGATE_PROXY_FUTURES.items()
                if fut.cancelled()
            ]
            for token in cancelled:
                _DELEGATE_PROXY_FUTURES.pop(token, None)
        if cancelled:
            try:
                await _request_parent_async(
                    "delegate_transformation_cancel", {"tokens": cancelled}
                )
            except Exception:
                pass
        if not tokens:
            continue
        for idx in range(0, len(tokens), _DELEGATE_POLL_BATCH):
            batch = tokens[idx : idx + _DELEGATE_POLL_BATCH]
            try:
                response = await _request_parent_async(
                    "delegate_transformation_poll", {"tokens": batch}
                )
            except Exception:
                continue
            if not isinstance(response, dict):
                continue
            for token, payload in response.items():
                with _DELEGATE_PROXY_LOCK:
                    fut = _DELEGATE_PROXY_FUTURES.pop(token, None)
                if fut is None:
                    continue
                status = payload.get("status") if isinstance(payload, dict) else None
                if status == "done":
                    result_hex = payload.get("result")
                    try:
                        fut.set_result(Checksum(result_hex))
                    except Exception:
                        fut.set_result(result_hex)
                elif status == "error":
                    err = payload.get("error") if isinstance(payload, dict) else "error"
                    if not isinstance(err, str):
                        err = repr(err)
                    fut.set_result(err)
                elif status == "cancelled":
                    fut.set_result("cancelled")
                elif status == "refused":
                    fut.set_result(_DELEGATION_REFUSED)
                elif status == "unknown":
                    fut.set_result("unknown")
                else:
                    fut.set_result("error")


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that creates daemon threads so exit is not blocked."""

    def _adjust_thread_count(self) -> None:  # type: ignore[override]
        if len(self._threads) < self._max_workers:
            thread_name = f"{self._thread_name_prefix}_{len(self._threads)}"

            def weakref_cb(_, q=self._work_queue):
                q.put(None)

            t = threading.Thread(
                name=thread_name,
                target=_cf_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
            )
            t.daemon = True
            t.start()
            self._threads.add(t)


_DASK_DELEGATE_MAX_WORKERS = min(64, (os.cpu_count() or 1) * 4)
_DASK_DELEGATE_EXECUTOR = _DaemonThreadPoolExecutor(
    max_workers=_DASK_DELEGATE_MAX_WORKERS,
    thread_name_prefix="dask-delegate",
)


def has_spawned() -> bool:
    """Return True if workers have been spawned in this process."""

    return _has_spawned


def dask_available() -> bool:
    """Return True if Dask submission is available in this process."""

    return _dask_available


def _set_has_spawned(value: bool) -> None:
    """Internal helper to reset spawn flag (testing/cleanup)."""

    global _has_spawned
    _has_spawned = bool(value)


def _set_dask_available(value: bool) -> None:
    """Internal helper to set the Dask availability flag."""

    global _dask_available
    _dask_available = bool(value)
    if _dask_available:
        os.environ[_DASK_AVAILABLE_ENV] = "1"
    else:
        os.environ.pop(_DASK_AVAILABLE_ENV, None)


def _request_parent_sync(op: str, payload: Any) -> Any:
    channel, loop = _require_child_channel()
    future = asyncio.run_coroutine_threadsafe(channel.request(op, payload), loop)
    return future.result()


async def _request_parent_async(op: str, payload: Any) -> Any:
    channel, loop = _require_child_channel()
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        return await channel.request(op, payload)
    future = asyncio.run_coroutine_threadsafe(channel.request(op, payload), loop)
    return await asyncio.wrap_future(future)


def _upload_buffer_to_parent(buf: Buffer, pointer: Dict[str, Any]) -> None:
    shm = SharedMemory(name=pointer["name"])
    try:
        view = shm.buf[: pointer["size"]]
        view[:] = buf.content
        del view
    finally:
        shm.close()
    _request_parent_sync("upload", {"pointer": pointer})


def _buffer_ref_op(buf: Buffer, op: str) -> None:
    if not len(buf):
        return
    checksum = buf.get_checksum()
    try:
        if op == "tempref" and len(buf.content) <= 1024 * 1024:
            _LOCAL_BUFFERS[checksum.hex()] = bytes(buf.content)
    except Exception:
        pass
    response = _request_parent_sync(
        "ref_op",
        {
            "op": op,
            "checksum": checksum.hex(),
            "length": len(buf.content),
        },
    )
    if response not in (None, 0):
        _upload_buffer_to_parent(buf, response)


def _buffer_incref(self: Buffer) -> None:
    _buffer_ref_op(self, "incref")


def _buffer_decref(self: Buffer) -> None:
    _buffer_ref_op(self, "decref")


def _buffer_tempref(self: Buffer, **_kwargs: Any) -> None:
    _buffer_ref_op(self, "tempref")


def _checksum_resolve(self: Checksum, celltype=None):
    data = _LOCAL_BUFFERS.get(self.hex())
    if data is not None:
        buf = Buffer(data, checksum=self)
        if celltype is None:
            return buf
        return buf.get_value(celltype)
    pointer = _request_parent_sync("download", {"checksum": self.hex()})
    if not isinstance(pointer, dict):
        raise CacheMissError(self)
    shm = SharedMemory(name=pointer["name"])
    try:
        data = bytes(shm.buf[: pointer["size"]])
    finally:
        shm.close()
    buf = Buffer(data, checksum=self)
    try:
        if celltype is None:
            return buf
        return buf.get_value(celltype)
    finally:
        _request_parent_sync("downloaded", {"pointer": pointer})


def _patch_worker_primitives() -> None:
    global _primitives_patched, _dummy_buffer_cache
    if _primitives_patched:
        return

    class _DummyBufferCache:
        def register(self, *args: Any, **kwargs: Any) -> None:
            return None

        def incref(self, *args: Any, **kwargs: Any) -> None:
            return None

        def decref(self, *args: Any, **kwargs: Any) -> None:
            return None

        def tempref(self, *args: Any, **kwargs: Any) -> None:
            return None

        def get(self, *args: Any, **kwargs: Any) -> None:
            return None

    _dummy_buffer_cache = _DummyBufferCache()

    def _worker_buffer_cache():
        return _dummy_buffer_cache

    # Monkeypatch buffer cache access to a dummy implementation.
    _buffer_cache.get_buffer_cache = _worker_buffer_cache  # type: ignore[assignment]
    _buffer_class.get_buffer_cache = _worker_buffer_cache  # type: ignore[assignment]

    # Checksum refcounting is disabled inside a worker.
    Checksum.incref = lambda self: None  # type: ignore[assignment]
    Checksum.decref = lambda self: None  # type: ignore[assignment]
    Checksum.tempref = (  # type: ignore[assignment]
        lambda self, interest=128.0, fade_factor=2.0, fade_interval=2.0: None
    )
    Buffer.incref = _buffer_incref  # type: ignore[assignment]
    Buffer.decref = _buffer_decref  # type: ignore[assignment]
    Buffer.tempref = _buffer_tempref  # type: ignore[assignment]
    Checksum.resolve = _checksum_resolve  # type: ignore[assignment]
    _primitives_patched = True


def _execute_transformation(payload: Dict[str, Any]) -> Checksum | str:
    return _execute_transformation_impl(payload, _format_pruned_exec_traceback)


def _execute_transformation_request(payload: Dict[str, Any]) -> Checksum | str:
    from . import SeamlessStreamTransformationError

    stdout_buffer = io.BytesIO()
    stderr_buffer = io.BytesIO()
    stdout_wrapper = io.TextIOWrapper(
        stdout_buffer, encoding="utf-8", write_through=True
    )
    stderr_wrapper = io.TextIOWrapper(
        stderr_buffer, encoding="utf-8", write_through=True
    )
    previous_stdout = sys.stdout
    previous_stderr = sys.stderr
    sys.stdout = stdout_wrapper
    sys.stderr = stderr_wrapper
    try:

        def _format_request_exc(exc: BaseException) -> str:
            if isinstance(exc, SeamlessStreamTransformationError):
                return str(exc)
            return _format_pruned_exec_traceback()

        result = _execute_transformation_impl(payload, _format_request_exc)
    finally:
        try:
            stdout_wrapper.flush()
            stderr_wrapper.flush()
        finally:
            sys.stdout = previous_stdout
            sys.stderr = previous_stderr
            try:
                stdout_wrapper.detach()
            except Exception:
                pass
            try:
                stderr_wrapper.detach()
            except Exception:
                pass
    if isinstance(result, str):
        stdout_text = stdout_buffer.getvalue().decode("utf-8", errors="replace")
        stderr_text = stderr_buffer.getvalue().decode("utf-8", errors="replace")
        if stdout_text:
            if not result:
                result = "\n"
            result += (
                "\n*************************************************\n"
                "* Standard output\n"
                "*************************************************\n"
                f"{stdout_text}\n"
                "*************************************************\n"
            )
        if stderr_text:
            if not result:
                result = "\n"
            result += (
                "\n*************************************************\n"
                "* Standard error\n"
                "*************************************************\n"
                f"{stderr_text}\n"
                "*************************************************\n"
            )
    return result


def _execute_transformation_impl(
    payload: Dict[str, Any],
    format_exc: Callable[[BaseException], str],
) -> Checksum | str:
    owner_dask_key = _normalize_owner_dask_key(payload.get("owner_dask_key"))
    previous_owner = _set_current_owner_dask_key(owner_dask_key)
    owner_dask_priority = _normalize_owner_dask_priority(
        payload.get("owner_dask_priority")
    )
    previous_priority = _set_current_owner_dask_priority(owner_dask_priority)
    try:
        tf_checksum = Checksum(payload["tf_checksum"])
        scratch = bool(payload.get("scratch", False))
        tf_dunder = payload.get("tf_dunder", {}) or {}
        transformation_dict = payload.get("transformation_dict")
        try:
            if (
                transformation_dict is None
                or "__code_text__" not in transformation_dict
            ):
                # Merge missing code text from the serialized transformation buffer if available.
                try:
                    tf_dict_resolved = tf_checksum.resolve(celltype="plain")
                except Exception:
                    tf_dict_resolved = None
                if isinstance(tf_dict_resolved, dict):
                    if transformation_dict is None:
                        transformation_dict = tf_dict_resolved
                    elif (
                        "__code_text__" not in transformation_dict
                        and "__code_text__" in tf_dict_resolved
                    ):
                        transformation_dict = dict(transformation_dict)
                        transformation_dict["__code_text__"] = tf_dict_resolved[
                            "__code_text__"
                        ]
        except Exception:
            pass
        try:
            if transformation_dict is None:
                transformation_dict = tf_checksum.resolve(celltype="plain")
            result = run_transformation_dict(
                transformation_dict, tf_checksum, tf_dunder, scratch
            )
            result_checksum = Checksum(result)
            result_checksum.tempref()
            return result_checksum
        except Exception as exc:
            return format_exc(exc)
    finally:
        _set_current_owner_dask_key(previous_owner)
        _set_current_owner_dask_priority(previous_priority)


def _format_pruned_exec_traceback() -> str:
    exc_type, exc, tb = sys.exc_info()
    if exc_type is None or exc is None:
        return traceback.format_exc()
    if tb is None:
        return "".join(traceback.format_exception_only(exc_type, exc))
    frames = traceback.extract_tb(tb)
    # Drop the outer frames so user tracebacks start at user code (exec_code is 4th frame).
    frames = frames[4:]
    if not frames:
        return "".join(traceback.format_exception_only(exc_type, exc))
    formatted = ["Traceback (most recent call last):\n"]
    formatted.extend(traceback.format_list(frames))
    formatted.extend(traceback.format_exception_only(exc_type, exc))
    return "".join(formatted)


async def _child_initializer(channel: ChildChannel) -> None:
    from seamless_config.extern_clients import set_remote_clients_from_env

    global _child_channel, _child_loop, _quiet
    _child_channel = channel
    _child_loop = asyncio.get_running_loop()
    set_is_worker(True)
    _patch_worker_primitives()
    set_remote_clients_from_env(include_dask=True)

    async def handle_execute(payload: Dict[str, Any]) -> Checksum | str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _execute_transformation_request, payload
        )

    async def handle_quiet(_payload: Any) -> str:
        _quiet = True
        if os.environ.get("SEAMLESS_DEBUG_TRANSFORMATION"):
            print("[worker] received quiet request", file=sys.stderr, flush=True)
        try:
            logging.getLogger(__name__).setLevel(logging.ERROR)
            logging.getLogger("seamless.transformer.process.channel").setLevel(
                logging.ERROR
            )
            logging.getLogger("seamless.transformer.process.manager").setLevel(
                logging.ERROR
            )
        except Exception:
            pass
        return "ok"

    channel.add_request_handler("execute_transformation", handle_execute)
    channel.add_request_handler("quiet", handle_quiet)
    if not channel.ready_notified:
        try:
            await channel.notify_ready({"role": "worker"})
        except ConnectionClosed:
            # Parent is already closing; exit quietly.
            try:
                channel.ready_notified = (
                    True  # prevent manager child_main from retrying
                )
            except Exception:
                pass
            return


class _WorkerManager:
    def __init__(self, worker_count: int) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop_runner, args=(self.loop,), daemon=True
        )
        self._thread.start()
        self._executor = _DaemonThreadPoolExecutor(thread_name_prefix="worker-manager")
        self.loop.call_soon_threadsafe(self.loop.set_default_executor, self._executor)
        self._manager = ProcessManager(
            _memory_provider,
            loop=self.loop,
            default_initializer=_child_initializer,
        )
        self._handles = []
        self._load: Dict[str, int] = {}
        self._pointers: Dict[str, _Pointer] = {}
        self._pointer_lock: Optional[asyncio.Lock] = None
        self._prefetched_buffers: Dict[str, bytes] = {}
        self._limits: Dict[str, asyncio.Semaphore] = {}
        self._delegate_registry: Dict[str, Dict[str, Any]] = {}
        self._delegate_completed: Dict[str, tuple[Dict[str, Any], float]] = {}
        self._delegate_lock: Optional[asyncio.Lock] = None
        self._delegate_cleanup_task: asyncio.Task | None = None
        self._delegate_owner_cancelled: Dict[str, float] = {}
        self._delegate_owner_by_handle: Dict[str, str] = {}

        init_future = asyncio.run_coroutine_threadsafe(
            self._async_init(worker_count), self.loop
        )
        init_future.result()

    def _loop_runner(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _async_init(self, worker_count: int) -> None:
        self._pointer_lock = asyncio.Lock()
        self._delegate_lock = asyncio.Lock()
        self._manager.add_parent_handler("download", self._handle_download)
        self._manager.add_parent_handler("downloaded", self._handle_downloaded)
        self._manager.add_parent_handler("ref_op", self._handle_ref_op)
        self._manager.add_parent_handler("upload", self._handle_upload)
        self._manager.add_parent_handler(
            "delegate_transformation_submit",
            self._handle_delegate_transformation_submit,
        )
        self._manager.add_parent_handler(
            "delegate_transformation_poll", self._handle_delegate_transformation_poll
        )
        self._manager.add_parent_handler(
            "delegate_transformation_cancel",
            self._handle_delegate_transformation_cancel,
        )
        self._manager.add_parent_handler(
            "delegate_transformation", self._handle_delegate_transformation
        )
        self._manager.add_parent_handler(
            "call_transformer_proxy", self._handle_call_transformer_proxy
        )
        for idx in range(worker_count):
            handle = await self._manager.start_worker(name=f"worker-{idx + 1}")
            self._handles.append(handle)
            self._load[handle.name] = 0
            self._limits[handle.name] = asyncio.Semaphore(TRANSFORMATION_THROTTLE)
        await asyncio.gather(*(h.wait_until_ready() for h in self._handles))
        self._delegate_cleanup_task = asyncio.create_task(self._delegate_cleanup_loop())

    def close(self, *, wait: bool = False) -> None:
        if _DEBUG_SHUTDOWN:
            print(
                f"[_WorkerManager.close] start wait={wait}",
                file=sys.stderr,
                flush=True,
            )

        def _run_coro(factory, timeout: float) -> None:
            try:
                coro = factory()
                if self.loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
                    fut.result(timeout=timeout)
                else:
                    asyncio.run(coro)
            except Exception:
                try:
                    coro.close()
                except Exception:
                    pass

        if _DEBUG_SHUTDOWN:
            print(
                "[_WorkerManager.close] shutting down manager",
                file=sys.stderr,
                flush=True,
            )
        _run_coro(lambda: self._manager.aclose(), 3.0 if wait else 0.5)
        if _DEBUG_SHUTDOWN:
            print(
                "[_WorkerManager.close] manager closed, shutting down asyncgens",
                file=sys.stderr,
                flush=True,
            )
        _run_coro(lambda: self.loop.shutdown_asyncgens(), 1.0 if wait else 0.2)
        if _DEBUG_SHUTDOWN:
            print(
                "[_WorkerManager.close] asyncgens closed, shutting down executor",
                file=sys.stderr,
                flush=True,
            )
        _run_coro(lambda: self.loop.shutdown_default_executor(), 1.0 if wait else 0.2)

        async def _drain_pending_tasks():
            current = asyncio.current_task(loop=self.loop)
            tasks = [
                task
                for task in asyncio.all_tasks(loop=self.loop)
                if task is not current and not task.done()
            ]
            if not tasks:
                return
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        _run_coro(lambda: _drain_pending_tasks(), 1.0 if wait else 0.5)
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        try:
            self._thread.join(timeout=2.0 if wait else 0.2)
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        if _DEBUG_SHUTDOWN:
            print("[_WorkerManager.close] finished", file=sys.stderr, flush=True)

    def _select_handle(self):
        if not self._handles:
            raise RuntimeError("No worker handles available")
        return min(self._handles, key=lambda h: (self._load.get(h.name, 0), h.name))

    async def _dispatch(
        self,
        transformation_dict: Dict[str, Any] | None,
        tf_checksum: Checksum,
        tf_dunder: Dict[str, Any],
        scratch: bool,
        *,
        enforce_limit: bool = True,
        owner_dask_key: str | None = None,
        owner_dask_priority: int | None = None,
    ) -> Checksum | str:
        await self._prefetch_transformation_assets(transformation_dict, tf_checksum)
        retry_attempts = 0
        max_retries = max(1, len(self._handles) * 3)
        while True:
            handles_sorted = sorted(
                self._handles,
                key=lambda h: (self._load.get(h.name, 0), h.name),
            )
            handle = None
            limit: asyncio.Semaphore | None = None
            acquired = False
            # Prefer an available semaphore; if all are saturated, wait until one frees up.
            while handle is None:
                for candidate in handles_sorted:
                    cand_limit = self._limits.get(candidate.name)
                    if (
                        not enforce_limit
                        or cand_limit is None
                        or not cand_limit.locked()
                    ):
                        handle = candidate
                        limit = cand_limit
                        break
                if handle is None:
                    await asyncio.sleep(0.01)
            if bool(os.environ.get("SEAMLESS_DEBUG_TRANSFORMATION")):
                if not hasattr(self, "_debug_handles_printed"):
                    print(
                        f"[dispatch] handles={[h.name for h in self._handles]}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._debug_handles_printed = True
                print(
                    f"[dispatch] selecting {getattr(handle, 'name', '?')} "
                    f"load={self._load.get(getattr(handle, 'name', None), 0)}",
                    file=sys.stderr,
                    flush=True,
                )
            await handle.wait_until_ready()
            if enforce_limit and limit is not None:
                await limit.acquire()
                acquired = True

            self._load[handle.name] = self._load.get(handle.name, 0) + 1
            if owner_dask_key:
                self._delegate_owner_by_handle[handle.name] = owner_dask_key
            retry = False
            result: Checksum | str | None = None
            try:
                payload = {
                    "transformation_dict": transformation_dict,
                    "tf_checksum": Checksum(tf_checksum),
                    "tf_dunder": tf_dunder,
                    "scratch": scratch,
                }
                if owner_dask_key is not None:
                    payload["owner_dask_key"] = owner_dask_key
                if owner_dask_priority is not None:
                    payload["owner_dask_priority"] = owner_dask_priority
                result = await handle.request("execute_transformation", payload)
            except ProcessError:
                retry = True
                retry_attempts += 1
                try:
                    handle.ready_event.clear()
                except Exception:
                    pass
                if retry_attempts >= max_retries:
                    raise
            finally:
                self._load[handle.name] = max(0, self._load.get(handle.name, 0) - 1)
                if (
                    owner_dask_key
                    and self._delegate_owner_by_handle.get(handle.name)
                    == owner_dask_key
                ):
                    self._delegate_owner_by_handle.pop(handle.name, None)
                if acquired and limit is not None:
                    limit.release()
            if not retry:
                assert result is not None
                return result
            await asyncio.sleep(0.05)

    async def run_transformation_async(
        self,
        transformation_dict: Dict[str, Any] | None,
        tf_checksum: Checksum,
        tf_dunder: Dict[str, Any],
        scratch: bool,
        owner_dask_key: str | None = None,
        owner_dask_priority: int | None = None,
    ) -> Checksum | str:
        fut = asyncio.run_coroutine_threadsafe(
            self._dispatch(
                transformation_dict,
                tf_checksum,
                tf_dunder,
                scratch,
                enforce_limit=True,
                owner_dask_key=owner_dask_key,
                owner_dask_priority=owner_dask_priority,
            ),
            self.loop,
        )
        return await asyncio.wrap_future(fut)

    def run_transformation_sync(
        self,
        transformation_dict: Dict[str, Any] | None,
        tf_checksum: Checksum,
        tf_dunder: Dict[str, Any],
        scratch: bool,
        owner_dask_key: str | None = None,
        owner_dask_priority: int | None = None,
    ) -> Checksum | str:
        fut = asyncio.run_coroutine_threadsafe(
            self._dispatch(
                transformation_dict,
                tf_checksum,
                tf_dunder,
                scratch,
                enforce_limit=True,
                owner_dask_key=owner_dask_key,
                owner_dask_priority=owner_dask_priority,
            ),
            self.loop,
        )
        return fut.result()

    async def _store_delegate_entry(self, token: str, entry: Dict[str, Any]) -> None:
        if self._delegate_lock is None:
            return
        async with self._delegate_lock:
            self._delegate_registry[token] = entry

    def _delegate_payload_from_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        futures = entry.get("futures")
        thin = getattr(futures, "thin", None) if futures is not None else None
        if thin is None:
            return {"status": "error", "error": "missing thin future"}
        try:
            if thin.cancelled():
                return {"status": "cancelled"}
            if not thin.done():
                return {"status": "cancelled"}
            tf_checksum_hex, result_checksum_hex, exc = thin.result()
        except Exception as exc:
            return {"status": "error", "error": repr(exc)}
        if exc:
            err = exc if isinstance(exc, str) else repr(exc)
            return {"status": "error", "error": err}
        if result_checksum_hex is None:
            return {"status": "error", "error": "Result checksum unavailable"}
        return {"status": "done", "result": result_checksum_hex}

    async def _get_delegate_completed(self, token: str) -> Dict[str, Any] | None:
        if self._delegate_lock is None:
            return None
        async with self._delegate_lock:
            item = self._delegate_completed.pop(token, None)
        if item is None:
            return None
        payload, timestamp = item
        if _DELEGATE_COMPLETED_TTL > 0:
            try:
                if (time.monotonic() - float(timestamp)) >= _DELEGATE_COMPLETED_TTL:
                    return None
            except Exception:
                return None
        return dict(payload)

    async def _update_delegate_owner_states(self) -> None:
        if self._delegate_lock is None:
            return
        async with self._delegate_lock:
            entries = list(self._delegate_registry.values())
        owner_groups: Dict[Any, set[str]] = {}
        for entry in entries:
            owner_key = entry.get("owner_dask_key")
            if owner_key is None:
                continue
            dask_client = entry.get("dask_client")
            raw_client = getattr(dask_client, "client", None)
            if raw_client is None:
                continue
            owner_groups.setdefault(raw_client, set()).add(owner_key)
        if not owner_groups:
            return
        loop = asyncio.get_running_loop()
        tasks = []
        for raw_client, keys in owner_groups.items():
            key_list = list(keys)
            if not key_list:
                continue
            tasks.append(
                loop.run_in_executor(
                    _DASK_DELEGATE_EXECUTOR,
                    _fetch_scheduler_task_states,
                    raw_client,
                    key_list,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        now = time.monotonic()
        alive: set[str] = set()
        cancelled: set[str] = set()
        debug_items: list[tuple[str, str | None, bool | None]] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            for key, info in result.items():
                state = None
                who_wants = None
                if isinstance(info, tuple) and len(info) == 2:
                    state, who_wants = info
                if _DELEGATE_DEBUG_CLEANUP:
                    debug_items.append((key, state, who_wants))
                if _owner_task_cancelled(state, who_wants):
                    cancelled.add(key)
                else:
                    alive.add(key)
        for key in alive:
            self._delegate_owner_cancelled.pop(key, None)
        for key in cancelled:
            self._delegate_owner_cancelled[key] = now
        if _DELEGATE_DEBUG_CLEANUP and debug_items:
            logger = logging.getLogger(__name__)
            try:
                summary = ", ".join(
                    f"{key}:{state}:{who}" for key, state, who in debug_items[:10]
                )
                logger.warning(
                    "[delegate-cleanup] owner-states alive=%s cancelled=%s sample=%s",
                    len(alive),
                    len(cancelled),
                    summary,
                )
            except Exception:
                pass
        if _DELEGATE_OWNER_CANCEL_TTL > 0:
            for key, timestamp in list(self._delegate_owner_cancelled.items()):
                try:
                    expired = (now - float(timestamp)) >= _DELEGATE_OWNER_CANCEL_TTL
                except Exception:
                    expired = True
                if expired:
                    self._delegate_owner_cancelled.pop(key, None)

    async def _is_owner_active(self, dask_client: Any, owner_key: str) -> bool:
        raw_client = getattr(dask_client, "client", None)
        if raw_client is None:
            return False
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                _DASK_DELEGATE_EXECUTOR,
                _fetch_scheduler_task_states,
                raw_client,
                [owner_key],
            )
        except Exception:
            return False
        info = result.get(owner_key)
        if info is None:
            return False
        state = None
        who_wants = None
        if isinstance(info, tuple) and len(info) == 2:
            state, who_wants = info
        return not _owner_task_cancelled(state, who_wants)

    async def _delegate_cleanup_loop(self) -> None:
        logger = logging.getLogger(__name__)
        while True:
            await asyncio.sleep(_DELEGATE_CLEANUP_INTERVAL)
            if self._delegate_lock is None:
                continue
            if _DELEGATE_DEBUG_CLEANUP:
                try:
                    missing_owner = 0
                    missing_owner_key = 0
                    for entry in self._delegate_registry.values():
                        if not entry.get("owner"):
                            missing_owner += 1
                        if not entry.get("owner_dask_key"):
                            missing_owner_key += 1
                    logger.warning(
                        "[delegate-cleanup] tick interval=%s registry=%s completed=%s missing_owner=%s missing_owner_key=%s",
                        _DELEGATE_CLEANUP_INTERVAL,
                        len(self._delegate_registry),
                        len(self._delegate_completed),
                        missing_owner,
                        missing_owner_key,
                    )
                except Exception:
                    pass
            await self._update_delegate_owner_states()
            now = time.monotonic()
            stale: list[
                tuple[str, Dict[str, Any], str | None, str | None, bool, bool, bool]
            ] = []
            async with self._delegate_lock:
                if _DELEGATE_COMPLETED_TTL > 0:
                    for token, item in list(self._delegate_completed.items()):
                        payload, timestamp = item
                        try:
                            expired_completed = (
                                now - float(timestamp)
                            ) >= _DELEGATE_COMPLETED_TTL
                        except Exception:
                            expired_completed = True
                        if expired_completed:
                            self._delegate_completed.pop(token, None)
                for token, entry in list(self._delegate_registry.items()):
                    owner = entry.get("owner")
                    owner_key = entry.get("owner_dask_key")
                    last_seen = entry.get("last_seen")
                    handle = (
                        self._manager._handles.get(owner) if owner else None
                    )  # type: ignore[attr-defined]
                    endpoint = getattr(handle, "endpoint", None) if handle else None
                    owner_closed = (
                        handle is None or endpoint is None or endpoint.is_closed()
                    )
                    expired = False
                    thin_state = "unknown"
                    if _DELEGATE_STALE_TTL > 0 and last_seen is not None:
                        try:
                            expired = (now - float(last_seen)) >= _DELEGATE_STALE_TTL
                        except Exception:
                            expired = False
                    futures = entry.get("futures")
                    thin = (
                        getattr(futures, "thin", None) if futures is not None else None
                    )
                    if thin is not None:
                        try:
                            if thin.cancelled():
                                thin_state = "cancelled"
                            elif thin.done():
                                thin_state = "done"
                            else:
                                thin_state = "pending"
                        except Exception:
                            thin_state = "error"
                    owner_cancelled = (
                        owner_key is not None
                        and owner_key in self._delegate_owner_cancelled
                    )
                    if owner_cancelled or owner_closed or expired:
                        stale.append(
                            (
                                token,
                                entry,
                                owner,
                                owner_key,
                                owner_closed,
                                expired,
                                owner_cancelled,
                            )
                        )
                        self._delegate_registry.pop(token, None)
            for (
                token,
                entry,
                owner,
                owner_key,
                owner_closed,
                expired,
                owner_cancelled,
            ) in stale:
                if owner_cancelled or owner_closed:
                    payload = {"status": "cancelled"}
                else:
                    payload = self._delegate_payload_from_entry(entry)
                if payload:
                    if self._delegate_lock is not None:
                        async with self._delegate_lock:
                            self._delegate_completed[token] = (payload, now)
                if _DELEGATE_DEBUG_CLEANUP:
                    last_seen_age = None
                    try:
                        if entry.get("last_seen") is not None:
                            last_seen_age = now - float(entry["last_seen"])
                    except Exception:
                        last_seen_age = None
                    logger.warning(
                        "[delegate-cleanup] cancel token=%s owner=%s owner_key=%s closed=%s expired=%s cancelled=%s last_seen_age=%.2fs thin=%s",
                        token,
                        owner,
                        owner_key,
                        owner_closed,
                        expired,
                        owner_cancelled,
                        -1.0 if last_seen_age is None else last_seen_age,
                        thin_state,
                    )
                self._release_delegate_entry(entry, cancel=True)

    async def _get_delegate_entry(self, token: str) -> Dict[str, Any] | None:
        if self._delegate_lock is None:
            return None
        async with self._delegate_lock:
            return self._delegate_registry.get(token)

    async def _pop_delegate_entry(self, token: str) -> Dict[str, Any] | None:
        if self._delegate_lock is None:
            return None
        async with self._delegate_lock:
            return self._delegate_registry.pop(token, None)

    def _release_delegate_entry(self, entry: Dict[str, Any], *, cancel: bool) -> None:
        futures = entry.get("futures")
        dask_client = entry.get("dask_client")
        permission_granted = entry.get("permission_granted")
        release_permission = entry.get("release_permission")
        if futures is not None:
            try:
                release = getattr(dask_client, "release_transformation_futures", None)
                if callable(release):
                    release(futures, cancel=cancel)
                else:
                    for fut in (futures.base, futures.thin, futures.fat):
                        if fut is None:
                            continue
                        raw_client = getattr(dask_client, "client", None)
                        try:
                            if (
                                raw_client is not None
                                and cancel
                                and not fut.cancelled()
                                and not fut.done()
                            ):
                                raw_client.cancel(fut, force=True)
                        except Exception:
                            pass
                        try:
                            fut.release()
                        except Exception:
                            pass
            except Exception:
                pass
        if permission_granted and release_permission is not None:
            try:
                release_permission()
            except Exception:
                pass

    async def _handle_delegate_transformation_submit(
        self, _handle, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        permission_granted = False
        release_permission = None
        try:
            transformation_dict = payload.get("transformation_dict")
            tf_checksum = Checksum(payload["tf_checksum"])
            tf_dunder = payload.get("tf_dunder", {}) or {}
            scratch = bool(payload.get("scratch", False))
            driver_context = _driver_flag_from_tf_dunder(tf_dunder)
            if not driver_context:
                driver_context = bool(payload.get("driver_context", False))
            owner_dask_key = _normalize_owner_dask_key(payload.get("owner_dask_key"))
            owner_dask_priority = _normalize_owner_dask_priority(
                payload.get("owner_dask_priority")
            )
            if not owner_dask_key:
                owner_dask_key = self._delegate_owner_by_handle.get(
                    getattr(_handle, "name", None) or ""
                )
            cached_result = await self._get_cached_transformation_result(tf_checksum)
            if cached_result is not None:
                return {"status": "done", "result": cached_result.hex()}
            await self._prefetch_transformation_assets(transformation_dict, tf_checksum)
            if transformation_dict is None:
                try:
                    resolved = tf_checksum.resolve(celltype="plain")
                except Exception:
                    resolved = None
                if isinstance(resolved, dict):
                    transformation_dict = resolved
            dask_client = None
            submission = None
            if isinstance(transformation_dict, dict):
                try:
                    from seamless_dask.transformer_client import (
                        get_seamless_dask_client,
                    )
                    from seamless_dask.types import (
                        TransformationInputSpec,
                        TransformationSubmission,
                    )
                except Exception:
                    dask_client = None
                else:
                    dask_client = get_seamless_dask_client()
                    if dask_client is None:
                        submission = None
                    else:
                        meta_payload = transformation_dict.get("__meta__", {}) or {}
                        allow_input_fingertip = bool(
                            meta_payload.get("allow_input_fingertip", False)
                        )
                        submission = TransformationSubmission(
                            transformation_dict=transformation_dict,
                            inputs={},
                            input_futures={},
                            tf_checksum=tf_checksum.hex(),
                            tf_dunder=tf_dunder,
                            scratch=scratch,
                            require_value=False,
                            allow_input_fingertip=allow_input_fingertip,
                        )
                        dep_checksums = _dependency_checksums_from_tf_dunder(tf_dunder)
                        inputs: Dict[str, TransformationInputSpec] = {}
                        input_futures: Dict[str, Any] = {}
                        for pinname, value in transformation_dict.items():
                            if pinname.startswith("__"):
                                continue
                            if not isinstance(value, tuple) or len(value) < 3:
                                continue
                            celltype, subcelltype, checksum_hex = value
                            dep_tf_checksum = dep_checksums.get(pinname)
                            dep_futures = None
                            if dep_tf_checksum:
                                get_futures = getattr(
                                    dask_client, "get_transformation_futures", None
                                )
                                if callable(get_futures):
                                    dep_futures = get_futures(dep_tf_checksum)
                            if dep_futures is not None:
                                if dep_futures.fat is None:
                                    if allow_input_fingertip:
                                        dep_futures.fat = (
                                            dask_client.ensure_fat_finger_future(
                                                dep_futures
                                            )
                                        )
                                    else:
                                        dep_futures.fat = dask_client.ensure_fat_future(
                                            dep_futures
                                        )
                                inputs[pinname] = TransformationInputSpec(
                                    name=pinname,
                                    celltype=celltype,
                                    subcelltype=subcelltype,
                                    checksum=None,
                                    kind="transformation",
                                )
                                input_futures[pinname] = dep_futures.fat
                                continue
                            if checksum_hex is None:
                                raise RuntimeError(
                                    f"Input '{pinname}' has no checksum or dependency future"
                                )
                            if dep_tf_checksum and dep_futures is None:
                                logging.getLogger(__name__).info(
                                    "[seamless-dask] delegate fallback to checksum pin=%s tf_checksum=%s",
                                    pinname,
                                    dep_tf_checksum,
                                )
                            if isinstance(checksum_hex, Checksum):
                                checksum_hex = checksum_hex.hex()
                            inputs[pinname] = TransformationInputSpec(
                                name=pinname,
                                celltype=celltype,
                                subcelltype=subcelltype,
                                checksum=checksum_hex,
                                kind="checksum",
                            )
                            if allow_input_fingertip:
                                input_futures[pinname] = (
                                    dask_client.get_fat_finger_checksum_future(
                                        checksum_hex
                                    )
                                )
                            else:
                                input_futures[pinname] = (
                                    dask_client.get_fat_checksum_future(checksum_hex)
                                )
                        submission.inputs = inputs
                        submission.input_futures = input_futures
            if dask_client is None or submission is None:
                has_capacity = False
                for handle in self._handles:
                    limit = self._limits.get(handle.name)
                    if limit is None or not limit.locked():
                        has_capacity = True
                        break
                if not has_capacity:
                    return {"status": "refused"}
                result = await self._dispatch(
                    transformation_dict,
                    tf_checksum,
                    tf_dunder,
                    scratch,
                    enforce_limit=True,
                    owner_dask_key=owner_dask_key,
                )
                if isinstance(result, Checksum):
                    try:
                        result.tempref()
                    except Exception:
                        pass
                    return {"status": "done", "result": result.hex()}
                return {"status": "error", "error": result}

            if (
                owner_dask_key is not None
                and owner_dask_key in self._delegate_owner_cancelled
            ):
                if dask_client is None or not await self._is_owner_active(
                    dask_client, owner_dask_key
                ):
                    return {"status": "cancelled"}
                self._delegate_owner_cancelled.pop(owner_dask_key, None)

            permission_denied = object()

            def _submit():
                nonlocal permission_granted, release_permission
                try:
                    from seamless_dask.permissions import (
                        try_request_permission,
                        release_permission as _release_permission,
                    )
                except Exception:
                    try_request_permission = release_permission = None
                else:
                    release_permission = _release_permission
                if try_request_permission is not None and not driver_context:
                    permission_granted = try_request_permission()
                    if not permission_granted:
                        return permission_denied
                priority_boost = (
                    owner_dask_priority
                    if owner_dask_priority is not None
                    else _DEFAULT_DASK_BASE_PRIORITY
                )
                return dask_client.submit_transformation(
                    submission, need_fat=False, priority_boost=priority_boost
                )

            loop = asyncio.get_running_loop()
            futures = await loop.run_in_executor(_DASK_DELEGATE_EXECUTOR, _submit)
            if futures is permission_denied:
                if permission_granted and release_permission is not None:
                    try:
                        release_permission()
                    except Exception:
                        pass
                return {"status": "refused"}
            token = uuid.uuid4().hex
            await self._store_delegate_entry(
                token,
                {
                    "futures": futures,
                    "dask_client": dask_client,
                    "permission_granted": permission_granted,
                    "release_permission": release_permission,
                    "owner": getattr(_handle, "name", None),
                    "owner_dask_key": owner_dask_key,
                    "last_seen": time.monotonic(),
                },
            )
            return {"status": "submitted", "token": token}
        except Exception:
            if permission_granted and release_permission is not None:
                try:
                    release_permission()
                except Exception:
                    pass
            return {"status": "error", "error": traceback.format_exc()}

    async def _handle_delegate_transformation_poll(
        self, _handle, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        tokens = []
        if isinstance(payload, dict):
            tokens = payload.get("tokens", []) or []
        results: Dict[str, Any] = {}
        now = time.monotonic()
        for token in tokens:
            entry = await self._get_delegate_entry(token)
            if entry is None:
                completed = await self._get_delegate_completed(str(token))
                if completed is not None:
                    results[str(token)] = completed
                else:
                    results[str(token)] = {"status": "unknown"}
                continue
            entry["last_seen"] = now
            futures = entry.get("futures")
            thin = getattr(futures, "thin", None) if futures is not None else None
            if thin is None:
                entry = await self._pop_delegate_entry(token)
                if entry is not None:
                    self._release_delegate_entry(entry, cancel=True)
                results[str(token)] = {
                    "status": "error",
                    "error": "missing thin future",
                }
                continue
            if thin.cancelled():
                entry = await self._pop_delegate_entry(token)
                if entry is not None:
                    self._release_delegate_entry(entry, cancel=True)
                results[str(token)] = {"status": "cancelled"}
                continue
            if not thin.done():
                continue
            try:
                _tf_checksum_hex, result_checksum_hex, exc = thin.result()
            except Exception as exc:
                entry = await self._pop_delegate_entry(token)
                if entry is not None:
                    self._release_delegate_entry(entry, cancel=True)
                results[str(token)] = {"status": "error", "error": repr(exc)}
                continue
            if exc:
                err = exc
                if not isinstance(err, str):
                    err = repr(err)
                status_payload = {"status": "error", "error": err}
            elif result_checksum_hex is None:
                status_payload = {
                    "status": "error",
                    "error": "Result checksum unavailable",
                }
            else:
                status_payload = {"status": "done", "result": result_checksum_hex}
            entry = await self._pop_delegate_entry(token)
            if entry is not None:
                self._release_delegate_entry(entry, cancel=True)
            results[str(token)] = status_payload
        return results

    async def _handle_delegate_transformation_cancel(
        self, _handle, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        tokens = []
        if isinstance(payload, dict):
            tokens = payload.get("tokens", []) or []
        cancelled: list[str] = []
        for token in tokens:
            entry = await self._pop_delegate_entry(token)
            if entry is None:
                if self._delegate_lock is not None:
                    async with self._delegate_lock:
                        self._delegate_completed.pop(str(token), None)
                continue
            self._release_delegate_entry(entry, cancel=True)
            cancelled.append(str(token))
        return {"cancelled": cancelled}

    async def _handle_delegate_transformation(
        self, _handle, payload: Dict[str, Any]
    ) -> Checksum | str:
        try:
            transformation_dict = payload.get("transformation_dict")
            tf_checksum = Checksum(payload["tf_checksum"])
            tf_dunder = payload.get("tf_dunder", {}) or {}
            scratch = bool(payload.get("scratch", False))
            driver_context = _driver_flag_from_tf_dunder(tf_dunder)
            if not driver_context:
                driver_context = bool(payload.get("driver_context", False))
            owner_dask_priority = _normalize_owner_dask_priority(
                payload.get("owner_dask_priority")
            )
            cached_result = await self._get_cached_transformation_result(tf_checksum)
            if cached_result is not None:
                try:
                    cached_result.tempref()
                except Exception:
                    pass
                return cached_result
            await self._prefetch_transformation_assets(transformation_dict, tf_checksum)
            if transformation_dict is None:
                try:
                    resolved = tf_checksum.resolve(celltype="plain")
                except Exception:
                    resolved = None
                if isinstance(resolved, dict):
                    transformation_dict = resolved
            dask_client = None
            if isinstance(transformation_dict, dict):
                try:
                    from seamless_dask.transformer_client import (
                        get_seamless_dask_client,
                    )
                    from seamless_dask.types import (
                        TransformationInputSpec,
                        TransformationSubmission,
                    )
                except Exception:
                    dask_client = None
                else:
                    dask_client = get_seamless_dask_client()
            if dask_client is not None:
                submission = TransformationSubmission(
                    transformation_dict=transformation_dict,
                    inputs={},
                    input_futures={},
                    tf_checksum=tf_checksum.hex(),
                    tf_dunder=tf_dunder,
                    scratch=scratch,
                    require_value=False,
                )

                permission_denied = object()

                def _submit():
                    permission_granted = False
                    futures = None
                    try:
                        from seamless_dask.permissions import (
                            try_request_permission,
                            release_permission,
                        )
                    except Exception:
                        try_request_permission = release_permission = None
                    log = logging.getLogger(__name__)

                    def _describe_raw_client(raw_client: Any) -> str:
                        if raw_client is None:
                            return "client=None"
                        scheduler_addr = None
                        try:
                            scheduler = getattr(raw_client, "scheduler", None)
                            scheduler_addr = getattr(scheduler, "address", None)
                        except Exception:
                            scheduler_addr = None
                        client_id = getattr(raw_client, "id", None)
                        status = getattr(raw_client, "status", None)
                        return (
                            "client_obj_id="
                            + str(id(raw_client))
                            + " client_id="
                            + str(client_id)
                            + " status="
                            + str(status)
                            + " scheduler="
                            + str(scheduler_addr)
                        )

                    if try_request_permission is not None and not driver_context:
                        permission_granted = try_request_permission()
                        if not permission_granted:
                            return permission_denied

                    try:
                        allow_input_fingertip = bool(
                            getattr(submission, "allow_input_fingertip", False)
                        )
                        if isinstance(transformation_dict, dict):
                            dep_checksums = _dependency_checksums_from_tf_dunder(
                                tf_dunder
                            )
                            inputs: Dict[str, TransformationInputSpec] = {}
                            input_futures: Dict[str, Any] = {}
                            for pinname, value in transformation_dict.items():
                                if pinname.startswith("__"):
                                    continue
                                if not isinstance(value, tuple) or len(value) < 3:
                                    continue
                                celltype, subcelltype, checksum_hex = value
                                dep_tf_checksum = dep_checksums.get(pinname)
                                dep_futures = None
                                if dep_tf_checksum:
                                    get_futures = getattr(
                                        dask_client, "get_transformation_futures", None
                                    )
                                    if callable(get_futures):
                                        dep_futures = get_futures(dep_tf_checksum)
                                if dep_futures is not None:
                                    if dep_futures.fat is None:
                                        if allow_input_fingertip:
                                            dep_futures.fat = (
                                                dask_client.ensure_fat_finger_future(
                                                    dep_futures
                                                )
                                            )
                                        else:
                                            dep_futures.fat = (
                                                dask_client.ensure_fat_future(dep_futures)
                                            )
                                    inputs[pinname] = TransformationInputSpec(
                                        name=pinname,
                                        celltype=celltype,
                                        subcelltype=subcelltype,
                                        checksum=None,
                                        kind="transformation",
                                    )
                                    input_futures[pinname] = dep_futures.fat
                                    continue
                                if checksum_hex is None:
                                    raise RuntimeError(
                                        f"Input '{pinname}' has no checksum or dependency future"
                                    )
                                if dep_tf_checksum and dep_futures is None:
                                    log.info(
                                        "[seamless-dask] delegate fallback to checksum pin=%s tf_checksum=%s",
                                        pinname,
                                        dep_tf_checksum,
                                    )
                                if isinstance(checksum_hex, Checksum):
                                    checksum_hex = checksum_hex.hex()
                                inputs[pinname] = TransformationInputSpec(
                                    name=pinname,
                                    celltype=celltype,
                                    subcelltype=subcelltype,
                                    checksum=checksum_hex,
                                    kind="checksum",
                                )
                                if allow_input_fingertip:
                                    input_futures[pinname] = (
                                        dask_client.get_fat_finger_checksum_future(
                                            checksum_hex
                                        )
                                    )
                                else:
                                    input_futures[pinname] = (
                                        dask_client.get_fat_checksum_future(checksum_hex)
                                    )
                            submission.inputs = inputs
                            submission.input_futures = input_futures
                        priority_boost = (
                            owner_dask_priority
                            if owner_dask_priority is not None
                            else _DEFAULT_DASK_BASE_PRIORITY
                        )
                        futures = dask_client.submit_transformation(
                            submission,
                            need_fat=False,
                            priority_boost=priority_boost,
                        )
                        try:
                            _tf_checksum_hex, result_checksum_hex, exc = (
                                futures.thin.result()
                            )
                        except BaseException as exc:
                            raw_client = getattr(dask_client, "client", None)
                            log.warning(
                                "[seamless-dask] delegate thin.result failed pid=%s thread=%s %s exc=%r",
                                os.getpid(),
                                threading.current_thread().name,
                                _describe_raw_client(raw_client),
                                exc,
                            )
                            raise
                        if exc:
                            return exc
                        if result_checksum_hex is None:
                            return "Result checksum unavailable"
                        return Checksum(result_checksum_hex)
                    finally:
                        if futures is not None:
                            try:
                                release = getattr(
                                    dask_client, "release_transformation_futures", None
                                )
                                if callable(release):
                                    release(futures, cancel=True)
                                else:
                                    for fut in (
                                        futures.base,
                                        futures.thin,
                                        futures.fat,
                                    ):
                                        if fut is None:
                                            continue
                                        raw_client = getattr(
                                            dask_client, "client", None
                                        )
                                        try:
                                            if (
                                                raw_client is not None
                                                and not fut.cancelled()
                                                and not fut.done()
                                            ):
                                                raw_client.cancel(fut, force=True)
                                        except Exception:
                                            pass
                                        try:
                                            fut.release()
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                        if permission_granted and release_permission is not None:
                            release_permission()

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(_DASK_DELEGATE_EXECUTOR, _submit)
                if result is not permission_denied:
                    if isinstance(result, Checksum):
                        try:
                            result.tempref()
                        except Exception:
                            pass
                    return result
            # try to find a handle with available throttle; refuse only if all are at cap
            result = None
            tried: set[str] = set()
            for _ in range(len(self._handles)):
                handle = self._select_handle()
                if handle.name in tried:
                    continue
                tried.add(handle.name)
                await handle.wait_until_ready()
                limit = self._limits.get(handle.name)
                if limit is not None and limit.locked():
                    continue
                acquired = False
                if limit is not None:
                    await limit.acquire()
                    acquired = True
                self._load[handle.name] = self._load.get(handle.name, 0) + 1
                try:
                    payload2 = {
                        "transformation_dict": transformation_dict,
                        "tf_checksum": tf_checksum,
                        "tf_dunder": tf_dunder,
                        "scratch": scratch,
                    }
                    result = await handle.request("execute_transformation", payload2)
                finally:
                    self._load[handle.name] = max(0, self._load.get(handle.name, 0) - 1)
                    if acquired and limit is not None:
                        limit.release()
                break
            else:
                return _DELEGATION_REFUSED
        except Exception:
            return traceback.format_exc()
        if isinstance(result, Checksum):
            try:
                result.tempref()
            except Exception:
                pass
        return result

    async def _handle_download(
        self, _handle, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        checksum = Checksum(payload["checksum"])
        data = self._prefetched_buffers.get(checksum.hex())
        if data is None:
            buf = checksum.resolve()
            assert isinstance(buf, Buffer)
            data = buf.content
        pointer = await self._allocate_pointer(
            len(data), {"checksum": checksum.hex(), "direction": "download"}
        )
        block = self._pointers[pointer["key"]]
        view = block.shm.buf
        try:
            view[: len(data)] = data
        finally:
            # Release the creator's handle; the shared segment stays alive until unlink.
            del view
            _close_shm(block.shm)
        return pointer

    async def _handle_downloaded(self, _handle, payload: Dict[str, Any]) -> None:
        pointer = payload.get("pointer") if isinstance(payload, dict) else payload
        if isinstance(pointer, dict):
            key = pointer.get("key")
        else:
            key = None
        if key is None:
            return
        await self._release_pointer(key)

    async def _handle_ref_op(self, _handle, payload: Dict[str, Any]) -> Any:
        op = payload["op"]
        checksum = Checksum(payload["checksum"])
        length = payload.get("length")
        if op == "decref":
            checksum.decref()
            return None
        if op == "incref":
            checksum.incref()
        elif op == "tempref":
            checksum.tempref()
        else:
            raise ValueError(op)
        if get_buffer_cache().get(checksum) is not None:
            return 0
        if length is None:
            raise ValueError("Buffer length is required for uploads")
        return await self._allocate_pointer(
            length,
            {"checksum": checksum.hex(), "length": length, "direction": "upload"},
        )

    async def _handle_upload(self, _handle, payload: Dict[str, Any]) -> Dict[str, str]:
        pointer = payload.get("pointer")
        if not pointer or "key" not in pointer:
            raise KeyError("Upload request missing pointer")
        block = await self._pop_pointer(pointer["key"])
        # Ensure the creator's handle does not hold an FD before re-opening to read.
        _close_shm(block.shm)
        reader = SharedMemory(name=block.shm.name)
        try:
            data = bytes(reader.buf[: block.size])
        finally:
            reader.close()
        block.shm.unlink()
        checksum_hex = block.metadata.get("checksum")
        checksum = Checksum(checksum_hex) if checksum_hex is not None else None
        buffer_obj = Buffer(data, checksum=checksum)
        checksum = buffer_obj.get_checksum()
        try:
            cache = get_buffer_cache()
            cache.register(checksum, buffer_obj, size=len(data))
            cache.incref(checksum, buffer_obj)
        except Exception:
            pass
        return {"checksum": checksum.hex()}

    async def _allocate_pointer(
        self, size: int, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if self._pointer_lock is None:
            raise RuntimeError("Pointer registry is not initialized")
        async with self._pointer_lock:
            key = uuid.uuid4().hex
            shm = SharedMemory(create=True, size=size)
            pointer = _Pointer(
                key=key, shm=shm, size=size, metadata=dict(metadata or {})
            )
            self._pointers[key] = pointer
            # For upload pointers we don't need to keep the creator's handle open.
            direction = pointer.metadata.get("direction")
            if direction == "upload":
                _close_shm(pointer.shm)
            return {
                "key": key,
                "name": shm.name,
                "size": size,
                "metadata": dict(pointer.metadata),
            }

    async def _release_pointer(self, key: str) -> None:
        if self._pointer_lock is None:
            return
        async with self._pointer_lock:
            block = self._pointers.pop(key, None)
        if block is None:
            return
        try:
            _close_shm(block.shm)
        finally:
            block.shm.unlink()

    async def _pop_pointer(self, key: str) -> _Pointer:
        if self._pointer_lock is None:
            raise RuntimeError("Pointer registry is not initialized")
        async with self._pointer_lock:
            block = self._pointers.pop(key)
        return block

    async def _handle_call_transformer_proxy(
        self, _handle, payload: Dict[str, Any]
    ) -> Any:
        proxy_id = payload.get("proxy_id")
        args = payload.get("args", ())
        kwargs = payload.get("kwargs", {})
        transformer_obj = _transformer_registry.get(proxy_id)
        if transformer_obj is None:
            raise KeyError(f"Unknown transformer proxy {proxy_id}")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, transformer_obj, *args, **kwargs)

    async def _prefetch_transformation_assets(
        self, transformation_dict: Dict[str, Any] | None, tf_checksum: Checksum
    ) -> None:
        if transformation_dict is None:
            return
        checksums: set[str] = set()
        checksums.add(Checksum(tf_checksum).hex())
        for key, value in transformation_dict.items():
            if isinstance(value, tuple) and len(value) >= 3:
                cs_hex = value[2]
                if _is_checksum_hex(cs_hex):
                    checksums.add(Checksum(cs_hex).hex())
            elif _is_checksum_hex(value):
                checksums.add(Checksum(value).hex())
        for cs_hex in checksums:
            if cs_hex in self._prefetched_buffers:
                continue
            try:
                buf = Checksum(cs_hex).resolve()
            except Exception:
                continue
            if isinstance(buf, Buffer):
                self._prefetched_buffers[cs_hex] = bytes(buf.content)

    async def _get_cached_transformation_result(
        self, tf_checksum: Checksum
    ) -> Checksum | None:
        try:
            from seamless_remote import database_remote
        except Exception:
            return None
        try:
            return await database_remote.get_transformation_result(tf_checksum)
        except Exception:
            return None


def spawn(num_workers: Optional[int] = None, dask_available: bool = False) -> None:
    global _worker_manager
    if not dask_available:
        try:
            if (
                mp.parent_process() is not None
                or mp.current_process().name != "MainProcess"
            ):
                print(
                    "seamless.transformer.worker.spawn() called outside MainProcess; ignoring",
                    file=sys.stderr,
                    flush=True,
                )
                return None
        except Exception:
            # If process inspection fails, fall back to attempting spawn.
            pass
    if _DEBUG_SHUTDOWN:
        print(
            f"[worker.spawn] entry has_spawned()={has_spawned()} manager={_worker_manager}",
            file=sys.stderr,
            flush=True,
        )
    if has_spawned():
        raise RuntimeError("Workers have already been spawned")
    ensure_open("spawn workers")
    worker_count = num_workers or (os.cpu_count() or 1)
    _set_dask_available(dask_available)
    if dask_available:
        os.environ.setdefault(_ALLOW_REMOTE_CLIENTS_ENV, "1")
    try:
        _worker_manager = _WorkerManager(worker_count)
    except Exception:
        _set_dask_available(False)
        raise
    _set_has_spawned(True)
    if _DEBUG_SHUTDOWN:
        print(
            f"[worker.spawn] spawned manager={_worker_manager} workers={worker_count}",
            file=sys.stderr,
            flush=True,
        )
    return None


def _require_manager() -> _WorkerManager:
    if _worker_manager is None:
        raise RuntimeError("Workers have not been spawned")
    return _worker_manager


def _cleanup_workers() -> None:
    global _worker_manager, _has_spawned
    if _worker_manager is None:
        return
    try:
        _worker_manager.close(wait=True)
    except Exception:
        pass
    _worker_manager = None
    _has_spawned = False
    _set_dask_available(False)


def shutdown_workers(*, wait: bool = True) -> None:
    """Explicitly shut down the worker pool and clear the spawn flag."""

    global _has_spawned, _worker_manager
    if _worker_manager is None:
        return
    try:
        _worker_manager.close(wait=wait)
    except Exception:
        pass
    _worker_manager = None
    _has_spawned = False
    _set_dask_available(False)


def get_throttle_load() -> tuple[int, int]:
    """Return (used_slots, total_slots) for the worker throttle semaphores."""

    manager = _worker_manager
    if manager is None:
        return 0, 0
    total = len(manager._limits) * TRANSFORMATION_THROTTLE  # type: ignore[attr-defined]
    used = 0
    for limit in manager._limits.values():  # type: ignore[attr-defined]
        current = getattr(limit, "_value", TRANSFORMATION_THROTTLE)
        used += max(0, TRANSFORMATION_THROTTLE - int(current))  # type: ignore[attr-defined]
    return used, total


async def dispatch_to_workers(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum: Checksum,
    tf_dunder: Dict[str, Any],
    scratch: bool,
    owner_dask_key: str | None = None,
    owner_dask_priority: int | None = None,
) -> Checksum | str:
    manager = _require_manager()
    return await manager.run_transformation_async(
        transformation_dict,
        tf_checksum,
        tf_dunder,
        scratch,
        owner_dask_key=owner_dask_key,
        owner_dask_priority=owner_dask_priority,
    )


async def forward_to_parent(
    transformation_dict: Dict[str, Any],
    *,
    tf_checksum: Checksum,
    tf_dunder: Dict[str, Any],
    scratch: bool,
) -> Checksum | str:
    payload = {
        "transformation_dict": transformation_dict,
        "tf_checksum": Checksum(tf_checksum),
        "tf_dunder": tf_dunder,
        "scratch": scratch,
    }
    owner_dask_key = _get_current_owner_dask_key()
    if owner_dask_key is not None:
        payload["owner_dask_key"] = owner_dask_key
    owner_dask_priority = _get_current_owner_dask_priority()
    if owner_dask_priority is not None:
        payload["owner_dask_priority"] = owner_dask_priority
    response = await _request_parent_async("delegate_transformation_submit", payload)
    if isinstance(response, dict):
        status = response.get("status")
        if status == "submitted":
            token = response.get("token")
            if not token:
                raise RuntimeError("Missing delegation token")
            proxy_future = _register_delegate_token(str(token))
            result = await asyncio.wrap_future(proxy_future)
            if isinstance(result, str):
                if result == _DELEGATION_REFUSED:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None, _execute_transformation, payload
                    )
                if result == "cancelled":
                    raise RuntimeError("Delegated transformation cancelled")
                if result == "unknown":
                    raise RuntimeError("Delegated transformation unknown")
                raise RuntimeError(result)
            if not isinstance(result, Checksum):
                raise RuntimeError(
                    f"Delegated transformation returned non-checksum result: {result!r}"
                )
            return result
        if status == "refused":
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _execute_transformation, payload)
        if status == "done":
            result_hex = response.get("result")
            if result_hex is None:
                raise RuntimeError("Delegated transformation returned no result")
            try:
                return Checksum(result_hex)
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid delegated checksum: {result_hex!r}"
                ) from exc
        if status == "error":
            raise RuntimeError(
                response.get("error") or "Delegated transformation error"
            )
    result = response
    if isinstance(result, str):
        if result == _DELEGATION_REFUSED:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _execute_transformation, payload)
        # any other string is an error/traceback
        raise RuntimeError(result)
    if not isinstance(result, (Checksum, bytes, bytearray)):
        raise RuntimeError(
            f"Delegated transformation returned non-checksum result: {result!r}"
        )
    return result


__all__ = [
    "spawn",
    "dispatch_to_workers",
    "forward_to_parent",
    "has_spawned",
    "dask_available",
    "get_throttle_load",
    "shutdown_workers",
]
