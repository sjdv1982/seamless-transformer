"""Parent-side process manager with async messaging and shared memory."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, Optional

from .channel import ChildChannel, Endpoint, ConnectionClosed
from .shared_memory import SharedMemoryRegistry
from .utils import run_handler


def _select_context() -> mp.context.BaseContext:
    """Select the process start method (always spawn)."""

    return mp.get_context("spawn")


_CTX = _select_context()


class ProcessError(RuntimeError):
    """Base class for process manager related failures."""


@dataclass
class ProcessHandle:
    manager: "ProcessManager"
    name: str
    initializer: Optional[Callable[[ChildChannel], Any]]
    restart: bool
    process: Optional[mp.Process] = None
    endpoint: Optional[Endpoint] = None
    pid: Optional[int] = None
    generation: int = 0
    closing: bool = False
    restarting: bool = False
    monitor_task: Optional[asyncio.Task[Any]] = None
    health_task: Optional[asyncio.Task[Any]] = None
    _condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)

    async def wait_for_generation(self, generation: int) -> None:
        async with self._condition:
            while self.generation < generation:
                await self._condition.wait()

    async def wait_until_ready(self) -> None:
        await self.ready_event.wait()

    def assign_runtime(self, process: mp.Process, endpoint: Endpoint) -> None:
        self.process = process
        self.endpoint = endpoint
        self.pid = process.pid
        self.ready_event.clear()

    async def mark_ready(self) -> None:
        async with self._condition:
            self.generation += 1
            self._condition.notify_all()
        self.ready_event.set()

    async def request(
        self, op: str, payload: Any = None, *, timeout: Optional[float] = None
    ) -> Any:
        endpoint = self.endpoint
        if endpoint is None or endpoint.is_closed():
            raise ProcessError(f"Worker {self.name} is not ready")
        return await endpoint.request(op, payload, timeout=timeout)

    def is_alive(self) -> bool:
        return bool(self.process and self.process.is_alive())

    def cancel_watchers(self) -> None:
        for task in (self.monitor_task, self.health_task):
            if task and not task.done():
                task.cancel()


class ProcessManager:
    """Coordinates worker processes and shared memory lifecycles."""

    def __init__(
        self,
        memory_provider: Callable[[str], Any],
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        health_check_interval: float = 1.0,
        health_check_timeout: float = 1.0,
        default_initializer: Optional[Callable[[ChildChannel], Any]] = None,
    ) -> None:
        self.loop = loop or asyncio.get_event_loop()
        self.memory_registry = SharedMemoryRegistry(memory_provider)
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self._default_initializer = default_initializer
        self._handles: Dict[str, ProcessHandle] = {}
        self._logger = logging.getLogger(__name__)
        self._closing = False
        self._parent_handlers: Dict[str, Callable[[ProcessHandle, Any], Any]] = {}
        self.add_parent_handler("__worker_ready__", self._handle_worker_ready)
        self.add_parent_handler("shm_incref", self._handle_incref)
        self.add_parent_handler("shm_decref", self._handle_decref)

    async def start_worker(
        self,
        *,
        name: Optional[str] = None,
        initializer: Optional[Callable[[ChildChannel], Any]] = None,
        restart: bool = True,
    ) -> ProcessHandle:
        worker_name = name or f"worker-{len(self._handles)+1}"
        if worker_name in self._handles:
            raise ValueError(f"Worker {worker_name!r} already exists")
        handle = ProcessHandle(
            manager=self,
            name=worker_name,
            initializer=initializer or self._default_initializer,
            restart=restart,
        )
        self._handles[worker_name] = handle
        await self._spawn_into_handle(handle)
        return handle

    def get_handle(self, name: str) -> ProcessHandle:
        return self._handles[name]

    def add_parent_handler(
        self, op: str, handler: Callable[[ProcessHandle, Any], Any]
    ) -> None:
        self._parent_handlers[op] = handler
        for handle in self._handles.values():
            if handle.endpoint:
                self._bind_parent_handler(handle, op, handler)

    async def aclose(self) -> None:
        if self._closing:
            return
        self._closing = True
        tasks = [self._stop_handle(handle) for handle in list(self._handles.values())]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.memory_registry.close()
        self._handles.clear()

    async def _stop_handle(self, handle: ProcessHandle) -> None:
        handle.closing = True
        handle.cancel_watchers()
        endpoint = handle.endpoint
        if endpoint:
            try:
                await handle.request("shutdown", None, timeout=1.0)
            except Exception:
                pass
            await endpoint.aclose()
        process = handle.process
        if process and process.is_alive():
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if handle.pid is not None:
            await self.memory_registry.reset_pid(handle.pid)
        handle.endpoint = None
        handle.process = None

    async def _spawn_into_handle(self, handle: ProcessHandle) -> None:
        parent_conn, child_conn = _CTX.Pipe()
        initializer = handle.initializer
        process = _CTX.Process(
            target=_worker_bootstrap,
            args=(child_conn, initializer),
            name=handle.name,
            daemon=True,
        )
        process.start()
        child_conn.close()
        endpoint = Endpoint(parent_conn, loop=self.loop, name=f"parent[{handle.name}]")
        for op, handler in self._parent_handlers.items():
            self._bind_parent_handler(handle, op, handler, endpoint)
        handle.assign_runtime(process, endpoint)
        handle.closing = False
        handle.restarting = False
        handle.cancel_watchers()
        handle.monitor_task = self.loop.create_task(self._watch_endpoint(handle))
        handle.health_task = None

    def _bind_parent_handler(
        self,
        handle: ProcessHandle,
        op: str,
        handler: Callable[[ProcessHandle, Any], Any],
        endpoint: Optional[Endpoint] = None,
    ) -> None:
        actual_endpoint = endpoint or handle.endpoint
        if not actual_endpoint:
            return

        async def wrapper(payload: Any, *, _handle=handle, _handler=handler) -> Any:
            return await run_handler(_handler, _handle, payload)

        actual_endpoint.add_request_handler(op, wrapper)

    async def _watch_endpoint(self, handle: ProcessHandle) -> None:
        endpoint = handle.endpoint
        if endpoint is None:
            return
        await endpoint.wait_closed()
        if self._closing or handle.closing or handle.restarting:
            return
        await self._handle_worker_failure(handle, "connection closed")

    async def _health_loop(self, handle: ProcessHandle) -> None:
        while not self._closing and not handle.closing:
            await asyncio.sleep(self.health_check_interval)
            if handle.closing or handle.restarting:
                return
            if not handle.process or not handle.process.is_alive():
                await self._handle_worker_failure(handle, "process exited")
                return
            try:
                await asyncio.wait_for(
                    handle.request("ping", None), self.health_check_timeout
                )
            except Exception:
                if handle.closing or handle.restarting:
                    return
                await self._handle_worker_failure(handle, "ping timeout")
                return

    async def _handle_worker_failure(self, handle: ProcessHandle, reason: str) -> None:
        if handle.restarting or handle.closing or self._closing:
            return
        if not handle.restart:
            # During shutdown or explicit disable, skip restart noise and just drop the handle.
            handle.restarting = False
            handle.cancel_watchers()
            if handle.endpoint:
                await handle.endpoint.aclose()
            if handle.process and handle.process.is_alive():
                handle.process.join(timeout=0.5)
            if handle.pid is not None:
                await self.memory_registry.reset_pid(handle.pid)
            self._handles.pop(handle.name, None)
            handle.endpoint = None
            handle.process = None
            return
        if not handle.ready_event.is_set():
            # Child never became ready; don't loop restarts.
            handle.restart = False
            handle.restarting = False
            self._handles.pop(handle.name, None)
            return
        self._logger.warning("Restarting worker %s (%s)", handle.name, reason)
        handle.restarting = True
        handle.cancel_watchers()
        if handle.endpoint:
            await handle.endpoint.aclose()
        if handle.process and handle.process.is_alive():
            handle.process.join(timeout=0.5)
        if handle.pid is not None:
            await self.memory_registry.reset_pid(handle.pid)
        if handle.restart:
            await self._spawn_into_handle(handle)
        else:
            handle.restarting = False
            self._handles.pop(handle.name, None)
            handle.endpoint = None
            handle.process = None

    async def _handle_worker_ready(
        self, handle: ProcessHandle, payload: Any
    ) -> Dict[str, Any]:
        if handle.closing or self._closing:
            return {"status": "closing"}
        pid = payload.get("pid") if isinstance(payload, dict) else None
        if pid is not None and pid != handle.pid:
            self._logger.warning(
                "Worker %s reported PID %s but handle tracked %s",
                handle.name,
                pid,
                handle.pid,
            )
            handle.pid = pid
        await handle.mark_ready()
        if handle.health_task is None or handle.health_task.done():
            handle.health_task = self.loop.create_task(self._health_loop(handle))
        return {"status": "ok", "generation": handle.generation}

    async def _handle_incref(
        self, handle: ProcessHandle, payload: Any
    ) -> Dict[str, Any]:
        if handle.pid is None:
            raise ProcessError("Worker PID not available")
        key = payload["key"]
        return await self.memory_registry.incref(key, handle.pid)

    async def _handle_decref(self, handle: ProcessHandle, payload: Any) -> None:
        if handle.pid is None:
            raise ProcessError("Worker PID not available")
        key = payload["key"]
        await self.memory_registry.decref(key, handle.pid)


def _worker_bootstrap(
    conn: Connection, initializer: Optional[Callable[[ChildChannel], Any]]
) -> None:
    asyncio.run(_child_main(conn, initializer))


async def _child_main(
    conn: Connection, initializer: Optional[Callable[[ChildChannel], Any]]
) -> None:
    loop = asyncio.get_event_loop()
    endpoint = Endpoint(conn, loop=loop, name=f"child[{os.getpid()}]")
    channel = ChildChannel(endpoint)

    async def _ping(_: Any) -> Dict[str, Any]:
        return {"pid": os.getpid()}

    async def _shutdown(_: Any) -> Dict[str, str]:
        loop.call_soon(endpoint.close)
        return {"status": "shutting_down"}

    channel.add_request_handler("ping", _ping)
    channel.add_request_handler("shutdown", _shutdown)
    if initializer is not None:
        try:
            await run_handler(initializer, channel)
        except Exception:  # pragma: no cover - child side logging
            logging.getLogger(__name__).exception(
                "Child initializer failed in PID %s", os.getpid()
            )
            loop.call_soon(endpoint.close)
            raise
    if not channel.ready_notified:
        try:
            await channel.notify_ready()
        except ConnectionClosed:
            return
    await channel.wait_closed()
