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
import os
import threading
import traceback
import uuid
import sys
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _worker as _cf_worker
import weakref
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Optional

import seamless.caching.buffer_cache as _buffer_cache
import seamless.buffer_class as _buffer_class
from seamless import Buffer, CacheMissError, Checksum, set_is_worker, ensure_open
from seamless.caching.buffer_cache import get_buffer_cache

from .process import ChildChannel, ProcessManager, ConnectionClosed
from .run import run_transformation_dict_in_process

_DASK_AVAILABLE_ENV = "SEAMLESS_DASK_AVAILABLE"
_DASK_SCHEDULER_ENV = "SEAMLESS_DASK_SCHEDULER"
_DASK_WORKERS_ENV = "SEAMLESS_DASK_WORKERS"
_DASK_REMOTE_CLIENTS_ENV = "SEAMLESS_DASK_REMOTE_CLIENTS"
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


def _configure_remote_clients_from_env() -> None:
    payload = os.environ.get(_DASK_REMOTE_CLIENTS_ENV)
    if not payload:
        return
    try:
        import json
        from seamless.config import set_remote_clients

        set_remote_clients(json.loads(payload))
    except Exception:
        pass


def _configure_dask_client_from_env() -> None:
    scheduler_address = os.environ.get(_DASK_SCHEDULER_ENV)
    if not scheduler_address:
        return
    try:
        from seamless_dask.transformer_client import (
            get_seamless_dask_client,
            set_seamless_dask_client,
        )
    except Exception:
        return
    if get_seamless_dask_client() is not None:
        return
    try:
        from distributed import Client as DistributedClient
        from seamless_dask.client import SeamlessDaskClient

        try:
            worker_count = int(os.environ.get(_DASK_WORKERS_ENV, "1") or 1)
        except Exception:
            worker_count = 1
        dask_client = DistributedClient(
            scheduler_address, timeout="10s", set_as_default=False
        )
        sd_client = SeamlessDaskClient(
            dask_client,
            worker_plugin_workers=worker_count,
        )
        set_seamless_dask_client(sd_client)
    except Exception:
        pass


def _request_parent_sync(op: str, payload: Any) -> Any:
    channel, loop = _require_child_channel()
    future = asyncio.run_coroutine_threadsafe(channel.request(op, payload), loop)
    return future.result()


async def _request_parent_async(op: str, payload: Any) -> Any:
    channel, loop = _require_child_channel()
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
    tf_checksum = Checksum(payload["tf_checksum"])
    scratch = bool(payload.get("scratch", False))
    tf_dunder = payload.get("tf_dunder", {}) or {}
    transformation_dict = payload.get("transformation_dict")
    try:
        if transformation_dict is None or "__code_text__" not in transformation_dict:
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
        result = run_transformation_dict_in_process(
            transformation_dict, tf_checksum, tf_dunder, scratch
        )
        result_checksum = Checksum(result)
        result_checksum.tempref()
        return result_checksum
    except Exception:
        return traceback.format_exc()


async def _child_initializer(channel: ChildChannel) -> None:
    global _child_channel, _child_loop, _quiet
    _child_channel = channel
    _child_loop = asyncio.get_running_loop()
    set_is_worker(True)
    _patch_worker_primitives()
    _configure_remote_clients_from_env()
    _configure_dask_client_from_env()

    async def handle_execute(payload: Dict[str, Any]) -> Checksum | str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _execute_transformation, payload)

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

        init_future = asyncio.run_coroutine_threadsafe(
            self._async_init(worker_count), self.loop
        )
        init_future.result()

    def _loop_runner(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _async_init(self, worker_count: int) -> None:
        self._pointer_lock = asyncio.Lock()
        self._manager.add_parent_handler("download", self._handle_download)
        self._manager.add_parent_handler("downloaded", self._handle_downloaded)
        self._manager.add_parent_handler("ref_op", self._handle_ref_op)
        self._manager.add_parent_handler("upload", self._handle_upload)
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
    ) -> Checksum | str:
        await self._prefetch_transformation_assets(transformation_dict, tf_checksum)
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
                if not enforce_limit or cand_limit is None or not cand_limit.locked():
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
        try:
            payload = {
                "transformation_dict": transformation_dict,
                "tf_checksum": Checksum(tf_checksum),
                "tf_dunder": tf_dunder,
                "scratch": scratch,
            }
            result = await handle.request("execute_transformation", payload)
        finally:
            self._load[handle.name] = max(0, self._load.get(handle.name, 0) - 1)
            if acquired and limit is not None:
                limit.release()
        return result

    async def run_transformation_async(
        self,
        transformation_dict: Dict[str, Any] | None,
        tf_checksum: Checksum,
        tf_dunder: Dict[str, Any],
        scratch: bool,
    ) -> Checksum | str:
        fut = asyncio.run_coroutine_threadsafe(
            self._dispatch(
                transformation_dict,
                tf_checksum,
                tf_dunder,
                scratch,
                enforce_limit=True,
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
    ) -> Checksum | str:
        fut = asyncio.run_coroutine_threadsafe(
            self._dispatch(
                transformation_dict,
                tf_checksum,
                tf_dunder,
                scratch,
                enforce_limit=True,
            ),
            self.loop,
        )
        return fut.result()

    async def _handle_delegate_transformation(
        self, _handle, payload: Dict[str, Any]
    ) -> Checksum | str:
        try:
            transformation_dict = payload.get("transformation_dict")
            tf_checksum = Checksum(payload["tf_checksum"])
            tf_dunder = payload.get("tf_dunder", {}) or {}
            scratch = bool(payload.get("scratch", False))
            await self._prefetch_transformation_assets(transformation_dict, tf_checksum)
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
                if cs_hex:
                    checksums.add(Checksum(cs_hex).hex())
            elif isinstance(value, str) and len(value) == 64:
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


def spawn(num_workers: Optional[int] = None, dask_available: bool = False) -> None:
    global _worker_manager
    if not dask_available:
        try:
            if mp.parent_process() is not None or mp.current_process().name != "MainProcess":
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
) -> Checksum | str:
    manager = _require_manager()
    return await manager.run_transformation_async(
        transformation_dict, tf_checksum, tf_dunder, scratch
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
    result = await _request_parent_async("delegate_transformation", payload)
    if isinstance(result, str):
        if result == _DELEGATION_REFUSED:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _execute_transformation, payload)
        # any other string is an error/traceback
        raise RuntimeError(result)
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
