import asyncio
import threading

import pytest
from seamless import Checksum

from seamless_transformer import transformation_cache
from seamless_transformer.transformation_cache import (
    TransformationCache,
    TransformationCancelledError,
)


def _checksum(char: str) -> Checksum:
    return Checksum(char * 64)


@pytest.fixture(autouse=True)
def no_remote_cache(monkeypatch):
    monkeypatch.setattr(transformation_cache, "database_remote", None)
    monkeypatch.setattr(transformation_cache, "buffer_remote", None)
    monkeypatch.setattr(transformation_cache, "jobserver_remote", None)
    monkeypatch.setattr(transformation_cache, "get_execution", lambda: "process")
    monkeypatch.setattr(transformation_cache, "is_worker", lambda: False)


def test_same_checksum_latches_onto_active_submission(monkeypatch):
    cache = TransformationCache()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_run(*_args):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return _checksum("b")

    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake_run)

    async def main():
        tf_checksum = _checksum("a")
        task1 = asyncio.create_task(
            cache.run(
                {"code": "return 1"},
                tf_checksum=tf_checksum,
                tf_dunder={"__meta__": {"local": True}},
                scratch=False,
                require_value=False,
                force_local=True,
            )
        )
        await asyncio.to_thread(started.wait, 5)
        task2 = asyncio.create_task(
            cache.run(
                {"code": "return 1"},
                tf_checksum=tf_checksum,
                tf_dunder={"__meta__": {"local": False}},
                scratch=False,
                require_value=False,
                force_local=True,
            )
        )
        await asyncio.sleep(0)
        assert cache.transformation_status(tf_checksum) == "running"
        release.set()
        assert await task1 == _checksum("b")
        assert await task2 == _checksum("b")

    asyncio.run(main())
    assert calls == [1]


def test_strict_different_dunder_rejects_while_active(monkeypatch):
    cache = TransformationCache()
    started = threading.Event()
    release = threading.Event()

    def fake_run(*_args):
        started.set()
        assert release.wait(5)
        return _checksum("d")

    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake_run)

    async def main():
        tf_checksum = _checksum("c")
        task = asyncio.create_task(
            cache.run(
                {"code": "return 1"},
                tf_checksum=tf_checksum,
                tf_dunder={"__meta__": {"local": True}},
                scratch=False,
                require_value=False,
                force_local=True,
            )
        )
        await asyncio.to_thread(started.wait, 5)
        with pytest.raises(RuntimeError, match="different dunder envelope"):
            await cache.run(
                {"code": "return 1"},
                tf_checksum=tf_checksum,
                tf_dunder={"__meta__": {"local": False}},
                scratch=False,
                require_value=False,
                force_local=True,
                strict_dunder=True,
            )
        release.set()
        assert await task == _checksum("d")

    asyncio.run(main())


def test_cancel_by_checksum_makes_strict_resubmission_possible(monkeypatch):
    cache = TransformationCache()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def fake_run(*_args):
        calls.append(1)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
            return _checksum("f")
        return _checksum("e")

    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake_run)

    async def main():
        tf_checksum = _checksum("e")
        task = asyncio.create_task(
            cache.run(
                {"code": "return 1"},
                tf_checksum=tf_checksum,
                tf_dunder={"__meta__": {"local": True}},
                scratch=False,
                require_value=False,
                force_local=True,
            )
        )
        await asyncio.to_thread(started.wait, 5)
        assert cache.cancel_by_checksum(tf_checksum) is True
        assert cache.cancel_by_checksum(tf_checksum) is False
        assert cache.transformation_status(tf_checksum) == "not-running"

        assert await cache.run(
            {"code": "return 1"},
            tf_checksum=tf_checksum,
            tf_dunder={"__meta__": {"local": False}},
            scratch=False,
            require_value=False,
            force_local=True,
            strict_dunder=True,
        ) == _checksum("e")

        release.set()
        with pytest.raises(TransformationCancelledError):
            await task

    asyncio.run(main())
    assert len(calls) == 2


def test_cancel_by_checksum_releases_owner_before_backend_finishes(monkeypatch):
    cache = TransformationCache()
    started = threading.Event()
    release = threading.Event()

    def fake_run(*_args):
        started.set()
        assert release.wait(5)
        return _checksum("8")

    monkeypatch.setattr(transformation_cache, "run_transformation_dict", fake_run)

    async def main():
        tf_checksum = _checksum("7")
        task = asyncio.create_task(
            cache.run(
                {"code": "return 1"},
                tf_checksum=tf_checksum,
                tf_dunder={"__meta__": {"local": True}},
                scratch=False,
                require_value=False,
                force_local=True,
            )
        )
        try:
            await asyncio.to_thread(started.wait, 5)
            assert cache.cancel_by_checksum(tf_checksum) is True
            with pytest.raises(TransformationCancelledError):
                await asyncio.wait_for(task, timeout=0.5)
        finally:
            release.set()

    asyncio.run(main())
