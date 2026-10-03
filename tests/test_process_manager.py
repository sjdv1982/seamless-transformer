"""Integration tests for the Seamless transformation process manager."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock
from typing import Any, Dict, List

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)

import pytest

from seamless_transformer.process import ChildChannel, MemoryPayload, ProcessManager
from seamless_transformer.worker import (
    _owner_task_cancelled,
    _worker_manager_executor_workers,
)


def run(coro):
    return asyncio.run(coro)


def make_manager(data_store: Dict[str, bytes]) -> ProcessManager:
    def provider(key: str) -> MemoryPayload:
        blob = data_store[key]
        return MemoryPayload(buffer=blob, metadata={"length": len(blob), "key": key})

    return ProcessManager(
        provider,
    )


def test_worker_manager_executor_has_room_for_pipe_readers() -> None:
    assert _worker_manager_executor_workers(1) == 32
    assert _worker_manager_executor_workers(40) == 88


def test_failure_watcher_can_finish_recovery() -> None:
    from seamless_transformer.process.manager import ProcessHandle

    async def check():
        sibling = asyncio.create_task(asyncio.sleep(60))
        handle = ProcessHandle(None, "test", None, True)
        handle.health_task = asyncio.current_task()
        handle.monitor_task = sibling
        handle.cancel_watchers()
        # Recovery must survive its next await; only the other watcher stops.
        await asyncio.sleep(0)
        assert sibling.cancelled()
        assert not handle.health_task.cancelling()

    run(check())


def test_health_check_accepts_traffic_but_restarts_silent_worker() -> None:
    async def check():
        endpoint = SimpleNamespace(received_messages=0, is_closed=lambda: False)
        attempts = 0

        async def ping(*_args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                endpoint.received_messages += 1
            await asyncio.Event().wait()

        handle = SimpleNamespace(
            closing=False,
            restarting=False,
            endpoint=endpoint,
            process=SimpleNamespace(is_alive=lambda: True),
            request=ping,
        )
        manager = make_manager({})
        manager.health_check_interval = 0.001
        manager.health_check_timeout = 0.01
        manager._handle_worker_failure = AsyncMock()
        await asyncio.wait_for(manager._health_loop(handle), 1)
        assert attempts == 2
        manager._handle_worker_failure.assert_awaited_once_with(handle, "ping timeout")
        await manager.aclose()

    run(check())


def test_delegate_input_preparation_keeps_manager_loop_responsive(monkeypatch) -> None:
    from seamless_dask import transformer_client
    from seamless_transformer.worker import _WorkerManager

    async def check():
        loop = asyncio.get_running_loop()
        responsive = threading.Event()

        def get_input(_checksum):
            loop.call_soon_threadsafe(responsive.set)
            if not responsive.wait(0.5):
                raise RuntimeError("manager loop blocked")
            raise RuntimeError("input preparation finished")

        client = SimpleNamespace(get_fat_checksum_future=get_input)
        monkeypatch.setattr(
            transformer_client, "get_seamless_dask_client", lambda: client
        )
        manager = _WorkerManager.__new__(_WorkerManager)
        manager._delegate_owner_by_handle = {}
        manager._get_cached_transformation_result = AsyncMock(return_value=None)
        manager._prefetch_transformation_assets = AsyncMock()
        result = await manager._handle_delegate_transformation_submit(
            SimpleNamespace(name="test"),
            {
                "tf_checksum": "01" * 32,
                "transformation_dict": {"a": ("plain", None, "02" * 32)},
            },
        )
        assert result["status"] == "error"
        assert "input preparation finished" in result["error"]

    run(check())


def test_processing_owner_task_is_alive_without_direct_waiter() -> None:
    assert not _owner_task_cancelled("processing", False)
    assert _owner_task_cancelled("released", True)
    assert _owner_task_cancelled(None, True)


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
