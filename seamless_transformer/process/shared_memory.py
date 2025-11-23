"""Shared memory reference counting for transformation workers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

from .utils import ensure_memoryview, run_handler


@dataclass
class MemoryPayload:
    """Data returned by a memory provider."""

    buffer: Any
    metadata: Mapping[str, Any] | None = None


@dataclass
class _Segment:
    key: str
    shm: SharedMemory
    size: int
    metadata: Dict[str, Any]
    refcounts: MutableMapping[int, int] = field(default_factory=dict)

    def total_refs(self) -> int:
        return sum(self.refcounts.values())

    def drop_pid(self, pid: int) -> None:
        self.refcounts.pop(pid, None)

    def close(self) -> None:
        self.shm.close()
        self.shm.unlink()


class SharedMemoryRegistry:
    """Tracks shared memory buffers requested by child processes."""

    def __init__(self, provider: Callable[[str], Any]) -> None:
        self._provider = provider
        self._segments: Dict[str, _Segment] = {}
        self._lock = asyncio.Lock()

    async def incref(self, key: str, pid: int) -> Dict[str, Any]:
        async with self._lock:
            segment = self._segments.get(key)
            if segment is None:
                payload = await run_handler(self._provider, key)
                if payload is None:
                    raise KeyError(f"Unknown shared memory key {key!r}")
                if not isinstance(payload, MemoryPayload):
                    payload = MemoryPayload(buffer=payload)
                segment = self._create_segment(key, payload)
                self._segments[key] = segment
            segment.refcounts[pid] = segment.refcounts.get(pid, 0) + 1
            return {
                "name": segment.shm.name,
                "size": segment.size,
                "metadata": dict(segment.metadata),
            }

    async def decref(self, key: str, pid: int) -> None:
        async with self._lock:
            segment = self._segments.get(key)
            if segment is None:
                raise KeyError(f"Key {key!r} not registered")
            count = segment.refcounts.get(pid)
            if not count:
                raise KeyError(f"PID {pid} has no reference to key {key!r}")
            if count == 1:
                segment.refcounts.pop(pid)
            else:
                segment.refcounts[pid] = count - 1
            if not segment.refcounts:
                segment.close()
                self._segments.pop(key, None)

    async def reset_pid(self, pid: int) -> None:
        async with self._lock:
            doomed = []
            for key, segment in self._segments.items():
                if pid in segment.refcounts:
                    segment.drop_pid(pid)
                    if not segment.refcounts:
                        doomed.append(key)
            for key in doomed:
                self._segments[key].close()
                self._segments.pop(key, None)

    async def close(self) -> None:
        async with self._lock:
            for segment in self._segments.values():
                segment.close()
            self._segments.clear()

    async def snapshot(self) -> Dict[str, Dict[str, Any]]:
        async with self._lock:
            result: Dict[str, Dict[str, Any]] = {}
            for key, segment in self._segments.items():
                result[key] = {
                    "size": segment.size,
                    "metadata": dict(segment.metadata),
                    "refcounts": dict(segment.refcounts),
                }
            return result

    def _create_segment(self, key: str, payload: MemoryPayload) -> _Segment:
        view = ensure_memoryview(payload.buffer)
        size = view.nbytes
        if size <= 0:
            raise ValueError(f"Cannot create empty shared memory block for {key!r}")
        shm = SharedMemory(create=True, size=size)
        shm.buf[:size] = view
        metadata = dict(payload.metadata or {})
        return _Segment(key=key, shm=shm, size=size, metadata=metadata)
