"""Async request/response plumbing for parent/child processes."""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Awaitable, Callable, Dict, Optional

from .utils import run_handler


class ConnectionClosed(RuntimeError):
    """Raised when a peer connection is closed."""


class Endpoint:
    """Bidirectional async request/response channel on top of a pipe."""

    def __init__(
        self,
        conn: Connection,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        name: Optional[str] = None,
    ) -> None:
        self._conn = conn
        self._loop = loop or asyncio.get_event_loop()
        self._logger = logging.getLogger(__name__)
        self._name = name or f"endpoint-{id(self):x}"
        self._pending: Dict[int, asyncio.Future[Any]] = {}
        self._handlers: Dict[str, Callable[[Any], Any]] = {}
        self._request_counter = itertools.count(1)
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._closed_event = asyncio.Event()
        self._request_tasks: set[asyncio.Task[Any]] = set()
        self._reader_task = self._loop.create_task(self._reader_loop())

    def add_request_handler(self, op: str, handler: Callable[[Any], Any]) -> None:
        self._handlers[op] = handler

    def remove_request_handler(self, op: str) -> None:
        self._handlers.pop(op, None)

    async def request(
        self, op: str, payload: Any = None, *, timeout: Optional[float] = None
    ) -> Any:
        if self._closed:
            raise ConnectionClosed(f"{self._name} is closed")
        request_id = next(self._request_counter)
        fut: asyncio.Future[Any] = self._loop.create_future()
        self._pending[request_id] = fut
        message = {"kind": "request", "op": op, "id": request_id, "payload": payload}
        try:
            await self._send(message)
        except Exception:
            self._pending.pop(request_id, None)
            raise
        try:
            if timeout is None:
                return await fut
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(request_id, None)

    async def _send(self, message: Any) -> None:
        async with self._send_lock:
            try:
                await self._loop.run_in_executor(None, self._conn.send, message)
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise ConnectionClosed(f"{self._name} send failed") from exc

    async def _reader_loop(self) -> None:
        try:
            while True:
                try:
                    message = await self._loop.run_in_executor(None, self._conn.recv)
                except (EOFError, OSError):
                    break
                except ConnectionClosed:
                    break
                await self._handle_message(message)
        except asyncio.CancelledError:
            pass
        finally:
            self._closed = True
            for pending in self._pending.values():
                if not pending.done():
                    pending.set_exception(ConnectionClosed(f"{self._name} closed"))
            self._pending.clear()
            for task in list(self._request_tasks):
                task.cancel()
            self._request_tasks.clear()
            try:
                self._conn.close()
            except OSError:
                pass
            self._closed_event.set()

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        kind = message.get("kind")
        if kind == "response":
            future = self._pending.get(message.get("id"))
            if not future or future.done():
                return
            if message.get("ok", True):
                future.set_result(message.get("payload"))
            else:
                future.set_exception(RuntimeError(message.get("error", "error")))
            return
        if kind != "request":
            self._logger.warning("%s received unknown message: %s", self._name, message)
            return
        op = message.get("op")
        request_id = message.get("id")
        self._logger.debug("%s scheduling request %s (%s)", self._name, request_id, op)
        self._spawn_request_task(request_id, op, message.get("payload"))

    def _spawn_request_task(self, request_id: Any, op: str, payload: Any) -> None:
        task = self._loop.create_task(self._process_request(request_id, op, payload))
        self._request_tasks.add(task)
        task.add_done_callback(self._request_tasks.discard)

    async def _process_request(self, request_id: Any, op: str, payload: Any) -> None:
        handler = self._handlers.get(op)
        if not handler:
            error = f"No handler for operation {op!r} on {self._name}"
            response = {
                "kind": "response",
                "id": request_id,
                "ok": False,
                "error": error,
            }
            await self._send(response)
            return
        try:
            result = await run_handler(handler, payload)
        except Exception as exc:  # pragma: no cover - best effort logging
            self._logger.exception("Handler %s on %s failed", op, self._name)
            response = {
                "kind": "response",
                "id": request_id,
                "ok": False,
                "error": repr(exc),
            }
        else:
            response = {
                "kind": "response",
                "id": request_id,
                "ok": True,
                "payload": result,
            }
        try:
            await self._send(response)
        except ConnectionClosed:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.close()
        except OSError:
            pass
        self._reader_task.cancel()
        for task in list(self._request_tasks):
            task.cancel()

    async def aclose(self) -> None:
        self.close()
        await self._closed_event.wait()

    def is_closed(self) -> bool:
        return self._closed

    async def wait_closed(self) -> None:
        await self._closed_event.wait()


@dataclass
class ChildSharedMemoryHandle:
    """Holds a shared memory buffer mapped by a child process."""

    channel: "ChildChannel"
    key: str
    name: str
    size: int
    metadata: Dict[str, Any]
    _shm: SharedMemory | None = None
    _closed: bool = False

    def __post_init__(self) -> None:
        self._shm = SharedMemory(name=self.name)

    @property
    def buffer(self) -> memoryview:
        if not self._shm:
            raise ConnectionClosed("Shared memory already released")
        return self._shm.buf[: self.size]

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        await self.channel.release_shared_memory(self.key)

    async def __aenter__(self) -> "ChildSharedMemoryHandle":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def close(self) -> None:
        if self._closed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
        else:
            loop.create_task(self.aclose())


class ChildChannel:
    """Utility wrapper that runs inside the child process."""

    def __init__(self, endpoint: Endpoint) -> None:
        self._endpoint = endpoint
        self._ready_notified = False

    def add_request_handler(self, op: str, handler: Callable[[Any], Any]) -> None:
        self._endpoint.add_request_handler(op, handler)

    async def request(
        self, op: str, payload: Any = None, *, timeout: Optional[float] = None
    ) -> Any:
        result = await self._endpoint.request(op, payload, timeout=timeout)
        if op == "__worker_ready__":
            self._ready_notified = True
        return result

    async def notify_ready(self, metadata: Optional[Dict[str, Any]] = None) -> Any:
        payload: Dict[str, Any] = {"pid": os.getpid()}
        if metadata:
            payload["metadata"] = metadata
        response = await self.request("__worker_ready__", payload)
        self._ready_notified = True
        return response

    @property
    def ready_notified(self) -> bool:
        return self._ready_notified

    async def acquire_shared_memory(
        self, key: str, *, timeout: Optional[float] = None
    ) -> ChildSharedMemoryHandle:
        response = await self.request(
            "shm_incref", {"key": key}, timeout=timeout
        )
        return ChildSharedMemoryHandle(
            channel=self,
            key=key,
            name=response["name"],
            size=response["size"],
            metadata=response.get("metadata", {}),
        )

    async def release_shared_memory(
        self, key: str, *, timeout: Optional[float] = None
    ) -> None:
        await self.request("shm_decref", {"key": key}, timeout=timeout)

    def close(self) -> None:
        self._endpoint.close()

    async def wait_closed(self) -> None:
        await self._endpoint.wait_closed()
