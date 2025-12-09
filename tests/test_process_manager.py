"""Integration tests for the Seamless transformation process manager."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

import pytest

from seamless.transformer.process import ChildChannel, MemoryPayload, ProcessManager


def run(coro):
    return asyncio.run(coro)


def make_manager(data_store: Dict[str, bytes]) -> ProcessManager:
    def provider(key: str) -> MemoryPayload:
        blob = data_store[key]
        return MemoryPayload(buffer=blob, metadata={"length": len(blob), "key": key})

    return ProcessManager(
        provider,
    )


def test_bidirectional_requests_and_shared_memory() -> None:
    run(_test_bidirectional_requests_and_shared_memory())


async def _test_bidirectional_requests_and_shared_memory() -> None:
    data = {"payload": (b"abc123" * 16)}
    manager = make_manager(data)
    try:

        async def add_handler(handle, payload):
            return payload["left"] + payload["right"]

        manager.add_parent_handler("add", add_handler)
        worker = await manager.start_worker(
            name="bidirectional", initializer=bidirectional_child_initializer
        )
        await worker.wait_until_ready()
        result = await worker.request("double", {"value": 21})
        assert result == 42
        shared = await worker.request("shared-len", {"key": "payload"})
        assert shared == len(data["payload"])
    finally:
        await manager.aclose()


def test_inputs_from_multiple_workers_share_memory() -> None:
    run(_test_inputs_from_multiple_workers_share_memory())


async def _test_inputs_from_multiple_workers_share_memory() -> None:
    data = {"buffer": os.urandom(2048)}
    manager = make_manager(data)
    try:
        handles = [
            await manager.start_worker(name="w1", initializer=shared_child_initializer),
            await manager.start_worker(name="w2", initializer=shared_child_initializer),
        ]
        await asyncio.gather(*(handle.wait_until_ready() for handle in handles))
        results = await asyncio.gather(
            *(handle.request("use-memory", {"key": "buffer"}) for handle in handles)
        )
        assert results == [len(data["buffer"]), len(data["buffer"])]
        snapshot = await manager.memory_registry.snapshot()
        assert "buffer" not in snapshot
    finally:
        await manager.aclose()


def test_refcounts_are_reset_when_worker_crashes() -> None:
    run(_test_refcounts_are_reset_when_worker_crashes())


async def _test_refcounts_are_reset_when_worker_crashes() -> None:
    data = {"blob": os.urandom(512)}
    manager = make_manager(data)
    try:
        worker = await manager.start_worker(
            name="unstable", initializer=crashing_child_initializer
        )
        await worker.wait_until_ready()
        await worker.request("touch-memory", {"key": "blob"})
        first_pid = worker.pid
        snapshot = await manager.memory_registry.snapshot()
        assert snapshot["blob"]["refcounts"].get(first_pid, 0) == 1
        current_generation = worker.generation
        with pytest.raises(RuntimeError):
            await worker.request("crash", None)
        await worker.wait_for_generation(current_generation + 1)
        assert worker.pid != first_pid
        snapshot = await manager.memory_registry.snapshot()
        assert "blob" not in snapshot
        await worker.request("touch-memory", {"key": "blob"})
    finally:
        await manager.aclose()


async def bidirectional_child_initializer(channel: ChildChannel) -> None:
    async def handle_double(payload: Dict[str, Any]) -> int:
        value = payload["value"]
        return await channel.request("add", {"left": value, "right": value})

    async def handle_shared(payload: Dict[str, Any]) -> int:
        handle = await channel.acquire_shared_memory(payload["key"])
        async with handle:
            return len(handle.buffer)

    channel.add_request_handler("double", handle_double)
    channel.add_request_handler("shared-len", handle_shared)


async def shared_child_initializer(channel: ChildChannel) -> None:
    async def handle_use_memory(payload: Dict[str, Any]) -> int:
        handle = await channel.acquire_shared_memory(payload["key"])
        async with handle:
            size = len(handle.buffer)
        return size

    channel.add_request_handler("use-memory", handle_use_memory)


async def crashing_child_initializer(channel: ChildChannel) -> None:
    async def handle_touch(payload: Dict[str, Any]) -> int:
        handle = await channel.acquire_shared_memory(payload["key"])
        return handle.buffer[0]

    async def handle_crash(_: Dict[str, Any]) -> None:
        os._exit(12)

    channel.add_request_handler("touch-memory", handle_touch)
    channel.add_request_handler("crash", handle_crash)
